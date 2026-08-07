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
| **Diffusion txt2img** (Krea 2) | ⚠️ ref | ✅ | ✅ | ✅ **primary** | cuda ≈ 1.42× ComfyUI gap; vulkan targets ~1.2× of cuda |
| **Diffusion txt2img** (SD1.5 / SDXL) | ⚠️ ref | ✅ | ✅ | ✅ **primary** | see §2A |
| **LLM text generation** | ⚠️ ref | ✅¹ | ✅ | ✅ **primary** | dispatched per-arch in `llm_main.zig` |
| **LLM vision (ViT/mmproj)** | ✅ | ⚠️ gemma3 only | ✅ | ✅ | see §5 |
| **GPU init failure** | — | → CPU fallback | → CPU fallback | → CPU fallback | logs + degrades, never hard-fails |

¹ vulkan LLM excludes **gemma4** entirely; qwen3 (incl. the llama/Mistral arch) runs GGUF block-quant weights on vulkan now — only a block-quant token *embedding* is rejected (no Vulkan gather kernel; see §4–§5).

---

## 2. Diffusion pipeline (`src/pipeline.zig`)

Per-stage dispatch order everywhere is `if (cu_be)` → CUDA, `else if (gpu_ctx)` → Vulkan, `else` → CPU.

Four families: krea2 (below), the SD family (§2A), and Z-Image (§2B).

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Text encoder** (Qwen3-VL-4B) | ✅ f32 | ✅ f32 (⚠️ f16 opt via `--encoder-f16`) | ✅ fp8→f16 TC | ✅ fp8→f16 TC | `krea2_text.zig`, `qwen3{,_gpu,_cuda}.zig` |
| **DiT** (Krea 2, 28 blocks) | ✅ all dtypes | ✅ fp8 / int8 / bf16 | ✅ fp8 / int8 / int4 / bf16 | ✅ fp8 / int8 / int4 / bf16 | `dit{,_gpu,_cuda}.zig` |
| **VAE decode** (Wan 2.1) | ✅ | ✅ | ✅ | ✅ (+cuDNN conv avail.) | `wan_vae.zig`, `vae_{gpu,cuda}.zig` |
| **VAE tiling** | CPU-tile | GPU-tile + CPU floor | GPU-tile + CPU floor | GPU-tile + CPU floor | `vae_tiled.zig` |
| **TAEHV preview** (taew2_1) | ✅ | ✅ *(new)* | ✅ | ✅ | `taehv{,_gpu,_cuda}.zig` |
| **latent2rgb preview** | ✅ | ✅ | ✅ | ✅ | `wan_vae.latentPreviewInto` (fallback when no `--taew`) |

**Cancellation** (`Options.cancel`): polled between sampling steps everywhere, plus mid-stage on every backend — between DiT blocks, between text-encoder layers, and between VAE decode layers (and per tile in `vae_tiled`). On the **cpu** backend the threaded matmul/attention kernels additionally poll a threadlocal token (`src/ops/cancel.zig`, armed by `dit.forward` / `wan_vae.Decoder.decode` / `qwen3.TextEncoder.encode`) per row-panel / k-block / query row, so a cancel lands in milliseconds even when a single CPU GEMM takes seconds. `error.Canceled` is never swallowed: the VAE OOM-retry ladder and the GPU→CPU encode fallback both propagate it.

### 2B. Z-Image ("zit" / `NextDiT`)

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Text encoder** (Qwen3-4B, penultimate) | ✅ f32 | ✅ bf16 TC | ⚠️ CPU (fp8-only arm) | ⚠️ CPU (fp8-only arm) | `zimage_text.zig`, `qwen3{,_gpu,_cuda}.zig` |
| **DiT** (`NextDiT`, 30 blocks + 2+2 refiners) | ✅ bf16 / f16 / f32 / fp8 / GGUF | ✅ bf16 / f16 / f32 / fp8 | ❌ CPU | ❌ CPU | `zimage{,_gpu}.zig` |
| **VAE decode** (`AutoencoderKL`, 16-ch) | ✅ | ⚠️ falls back to CPU | ⚠️ untested | ⚠️ untested | `sd_vae{,_gpu,_cuda}.zig` (`sd_vae.flux`) |
| **latent2rgb preview** | ✅ | ✅ | ✅ | ✅ | `zimage.latentPreviewInto` (Flux factors + bias) |

**Measured** (`-Dintegration -Dtest-filter="Z-Image gpu"`, needs `testdata/gpu-tests`):
`attn_full` at Z-Image's 30 x 128 head geometry matches `ops.attention` at **4.9e-7**
rel L2; the whole Vulkan forward matches `zimage.DiT.predict` on real checkpoint
weights at **2.1e-4** — the f16 tensor-core regime `sd_unet_cuda` already sits in.
Verified end to end: a Vulkan render is visually identical to the CPU one.

**Measured, full pipeline, 1056x1584 / 9 steps / cfg 1 / euler+beta** (RTX 3090):

| backend | s/step | decode | total | vs a real ComfyUI render |
|---|---|---|---|---|
| `cuda` | **2.94** | 11.9 s | **41.3 s** | **29.42 dB** / SSIM 0.959 |
| `vulkan` | 4.55 | 2.1 s | 50.5 s | 28.96 dB / SSIM 0.960 |

The two arms agree with each other at **34.20 dB / SSIM 0.986**, and both sit inside
ComfyUI's own **24.49 dB** bf16-vs-fp16 envelope — take that control before reading
either figure, because Z-Image Turbo at 9 steps is precision-dominated.

#### Text encoders: dtype x backend (measured 2026-08-06, 3090, 72-token prompt)

`--text-encoder` accepts **safetensors or GGUF**, opened by magic. Encode wall time,
and rel L2 against that same file's CPU forward (`TensorPencil te-test`):

| encoder weights | cpu | cuda | zig-cuda | vulkan |
|---|---|---|---|---|
| q4_k (`Qwen3-4B-Q4_K_M.gguf`, llama.cpp-style) | 1354 ms | **144 ms** / 1.3e-4 | 221 ms / 1.3e-4 | 947 ms / 1.3e-4 |
| q8_0 (`qwen_3_4b-q8_0.gguf`, ComfyUI-style) | 919 ms | 151 ms / 1.4e-4 | 210 ms / 1.5e-4 | 1010 ms / 1.5e-4 |
| bf16 (`qwen_3_4b.safetensors`) | 793 ms | 136 ms / 3.7e-3 | 187 ms / 3.7e-3 | 127 ms / 3.7e-3 |
| fp8-e4m3 (krea2 Qwen3-VL) | 1628 ms | 161 ms / 1.3e-4 | 207 ms / 1.4e-4 | 384 ms / 1.4e-6 |

Routing: fp8 -> `opMatmulFp8`; bf16 -> `opGemmBf16` (Ampere+, native bf16 tensor
cores, every encoder width satisfies `co%128==0 and k%32==0`) or `opMatmulBf16`;
block quants -> `opMatmulQuant` (CUDA) / `opMatmulCoopQuant` (Vulkan), i.e. the LLM's
own prefill paths, which suit an encoder exactly — one call over the whole prompt, so
the per-call weight dequant amortizes.

⚠️ **Block quants do NOT take the MMQ path** even where `mmqPipeFaster` says yes.
MMQ quantizes the activation to q8_1 (~0.5% rel); a conditioning tensor is computed
once per render and then steers every sampling step, so that trade sits the wrong way
round here. The LLM takes it because there the same GEMM runs per token, forever.

⚠️ **Vulkan's `wcode` maps every non-f8/bf16 dtype to `.f32`.** A block quant reaching
`qwen3_gpu.gemm`'s fallback would be read as f32 — the same silent-garbage shape as the
bf16 bug that file's `supportsWeights` comment records. The block-quant arm is checked
*before* `wcode`, and `supportsWeights` gates on `hasQuantPrefillGemm`.

**Why it is worth having, and which one to pick** (Z-Image, 1056x1584, seed 80085):

| encoder | file | live VRAM | vs ComfyUI | vs the bf16-encoder render | conditioning rel L2 |
|---|---|---|---|---|---|
| bf16 | 8.04 GB | 6738 MB | 29.4 dB | - | - |
| **q8_0** | 4.27 GB | **3634 MB** | **30.0 dB** | **30.8 dB** | 6.4e-3 |
| q4_k | 2.50 GB | 2065 MB | 21.8 dB | 22.2 dB | 3.4e-2 |

⚠️ **These are not the same trade and only a render shows it.** q8_0 lands INSIDE this
model's own 24.49 dB bf16-vs-fp16 floor — indistinguishable from bf16 at half the size.
q4_k is 8 dB below it: a different, equally valid image of the same prompt, not a
cheaper route to the same one. Recommend q8_0.

#### Where Z-Image's CUDA time goes (measured 2026-08-06, 3090, 1056x1584 / 9 steps)

| | s/step (avg, incl. warm-up) | steady state | step 1 |
|---|---|---|---|
| **ComfyUI** (bf16, torch + cuBLAS) | **1.836** | - | - |
| TensorPencil `cuda`, at first measurement | 3.17 | 2.6 | 8.0 s |
| + one-block-ahead `prefetchWeight` | 3.04 | 2.6 | 6.6 s |
| + coalesced `qk_rmsnorm_warp` | 2.39 | 2.0 | 5.6 s |
| + null-bias GEMM writing straight to `dst` | 2.27 | 1.9 | 5.1 s |
| + **parallel staging fill** | **2.00** | **1.8** | 3.4 s |

**91.8% of ComfyUI**, from 58%. Renders are unchanged in kind: 29.38 dB / SSIM 0.9592
against a real ComfyUI render of the same workflow, where the starting point was 29.42
/ 0.9586 — and ComfyUI's own bf16-vs-fp16 floor on this model is **24.49 dB**, so read
that control first.

⚠️ **The baseline is MEASURED under identical settings, not taken from a round
number.** `tools/render_zimage_ref.py` times `comfy.sample.sample` itself; "about 2
s/step" is really 1.836, which moved the target from "4% away" to "24% away".

##### The isolation, and why the first profile pointed at the wrong thing

`TensorPencil zimage-cuda-bench [<seq>] [libs]` replays exactly one step's device work
at Z-Image's own shapes, with no checkpoint and no sampler, split three ways. ⚠️ **It
exists because the whole-render profile was actively misleading**: it attributed 88% of
the step to `matmul`, but `opGemmBf16` is `ptic()`-scoped, so the two f32<->bf16
streaming passes wrapped around every GEMM were being counted *as* GEMM time — and the
norms hid inside a 7% `elt` bucket that looked too small to matter.

| | before | after | |
|---|---|---|---|
| GEMMs (224, with conversion) | 1393 ms | **1280 ms** | 65.2 TFLOP/s pure GEMM = ~92% of the card's bf16 roof |
| attention (32 x cuDNN SDPA) | 424 ms | 416 ms | 23.7 TFLOP/step at 57 TFLOP/s |
| elementwise | **765 ms** | **156 ms** | |
| total device work | 2556 ms | **1852 ms** | vs 1.8 s steady in a real render |

⚠️ **The GEMMs were never the problem, and a fourth of the step was two norm calls.**
Per-op bandwidth is what says so — these kernels do almost no arithmetic, so anything
far under ~936 GB/s is a kernel defect, not a workload:

| op (x32 blocks) | before | after |
|---|---|---|
| `qkNorm` q,k (hd=128) | 364 ms @ **37.5 GB/s** | 20 ms @ 698 GB/s |
| `qkNorm` sandwich (w=3840) | 290 ms @ **47.1 GB/s** | 29 ms @ 468 GB/s |
| `rmsMod` x2 | 27 ms @ 498 GB/s | unchanged |
| `rope` q,k / `gatedAdd` x2 / `siluMul` | 20 / 28 / 36 ms @ 684-749 GB/s | unchanged |

Every other kernel in the same block was already at 500-750 GB/s, which is what makes
the two outliers legible. Cause: `qk_rmsnorm` gave each **thread** a whole row, so a
warp's 32 loads were 32 sector fetches `4*dim` apart. `qk_rmsnorm_warp` gives each
**warp** a row (butterfly-shuffle reduction, no shared memory) - 17x and 10x. ⚠️ The
row sum becomes a tree where it was serial, so this is not bit-identical; it is the
more accurate of the two, and `rms_mod_par` next door already reduced this way.
`rows < 512` still takes the block-per-row kernel (LLM decode norms 1 x 2560).

##### Three more, each with its own receipt

- **A null bias skips a whole output pass.** `opGemmBf16(..., bias: ?[]const f32)`:
  with nothing to add and cuBLASLt free of the hand-PTX `m % 128` tiling rule, the
  GEMM writes f32 straight into `dst` - no `m` padding, no `conv_c` staging, no
  `bias_compact` re-read of `m*co` floats. Conversion overhead 143 -> 56 ms.
  ⚠️ Callers with a real bias (`dit_cuda`) keep the staged path untouched.
- ⚠️ **Weight upload is HOST-fill bound, not PCIe bound, and nothing said so until it
  was split out.** `TP_WARMUP_PROFILE` now reports the two halves separately: at
  11.57 GB it read `fill 4.88s, slot-wait 0.04s` - PCIe idle 99% of the time while one
  thread filled the pinned slots at **2.37 GB/s**, against 4.3 GB/s cold and 26 GB/s
  page-cached for the same file under `dd`. `Context.FillPool` fans the fill over 4
  threads (positional reads need no ordering): fill 4.88 -> 2.28 s, step 1 5.1 -> 2.6 s,
  and the render is **bit-identical** (PSNR ∞), as pure plumbing must be.
- **One-block-ahead `prefetchWeight`**, which `dit_cuda` had and `zimage_cuda` did not.

##### What is NOT the lever, so nobody re-walks it

- **cuBLASLt is at ~92% of the bf16 roof** at these shapes (65.2 TFLOP/s pure GEMM on a
  ~71 TFLOP/s card). Measured with `bench_gemm_only`, which removes exactly the
  conversion passes and nothing else.
- ⚠️ **`COMPUTE_32F` is not the problem.** Torch uses it too, so reduced-precision
  accumulate is not where ComfyUI's speed comes from; switching to `COMPUTE_16F` would
  change the numerics for a win the reference does not take.
- **M padding is 1.4%** at this size, and now zero on the null-bias path.
- Still untried, and now worth less than it looked: fusing the three qkv GEMMs into one
  `[11520, 3840]` (blocked on a de-interleave kernel - `qkNorm`/`rope`/`attn` all
  assume contiguous `[rows][heads*hd]`).

##### Validation

`TensorPencil zimage-cuda-test [<ckpt>] [libs]` checks `zimage_cuda.forward` against
`zimage.DiT.predict` on real weights (2 trunk layers), on **both** attention paths, and
exits non-zero on failure - the check this file's convention demanded and that
`zimage_cuda` had never had. Measured: `libs` 2.23e-4 / 2.14e-4, `hand_ptx` 2.37e-4 /
2.14e-4 rel L2, matching Vulkan's 2.19e-4 / 2.15e-4.

⚠️ **Found, NOT caused, while doing this: `cuda-dit-test` on a bf16 krea2 checkpoint
fails** at rel L2 0.141 and runs a 256px forward at ~11.5 s/step. A/B'd across the
`qkNorm` change (0.14061 old kernel vs 0.14098 new), so it is pre-existing and
independent of everything above. Unfixed.

⚠️ **The CUDA decode is 5x slower than Vulkan's** (11.9 s vs 2.1 s) and that is the
attention, not the convolutions: `sd_vae_cuda` uses `be.attn`'s f32 online softmax,
which is one thread per query — fine at a 64² tile, poor at a whole-image 132x198
latent (26136 single-head queries). Vulkan's f32 scores plane parallelizes better.
The fix is a *tiled* f32 online softmax or forcing the decode ladder to tile here;
neither is done.

⚠️ **`zimage_cuda` has no unit test.** Per this file's own convention the CUDA kernels
are checked by a CLI command (`sd-cuda-test`) because the test binary brings up no CUDA
context — a `zimage-cuda-test` is owed. What it *is* checked by today: the render
comparisons above, plus the isolation that localized its one bug (forcing the trunk to
CPU showed the fault was in the VAE, not the trunk).

⚠️ **The CUDA text-encoder arm is fp8-only** (`qwen3_cuda.encode` calls `opMatmulFp8`
unconditionally) and Z-Image's `qwen_3_4b.safetensors` is **bf16**, which it would read
as fp8 bytes and turn into noise. It gates on `qwen3_cuda.supportsWeights` and falls
back to the CPU encode. Widening it to bf16 is the cheapest real win available here.

⚠️ **The Vulkan text encoder had the same bug and it cost a blank white render**
(fixed 2026-08-06). `qwen3_gpu.gemm` handled fp8 only; a bf16 weight fell through to
`opMatmul` with `dtype_f8 = false`, i.e. the **f32** pipeline reading bf16 bytes. Every
GEMM was garbage, the conditioning non-finite, and the image solid white with no error.
⚠️ The trap worth remembering: this file's `wcode` helper *does* read bf16 natively —
but that is the **LM's GEMV** path, not the encoder's GEMM, and one comment does not
cover the other. `qwen3_gpu.supportsWeights` now gates it and `zimage_text` has a
GPU-vs-CPU parity test.

⚠️ **`sd_vae_gpu`'s mid-block scores plane is f32, not the f16 the tensor-core path
would give, and that is forced by measurement.** The Flux/Z-Image VAE's attention
**logits reach 9.95e6** on a 12x10 latent where SD1.5's reach **8.3** — 152x past
f16's 65504 ceiling, and even without overflow f16's quantum up there is ~8000, which
destroys a softmax whose differences are O(1). It rendered solid white with no error.
⚠️ Everything upstream is finite and in fact *smaller* than SD's (489 vs 929 at the
last level), so nothing but the logits themselves shows it — and the 4-channel-only
test could not, because 4 was also the hardcoded channel count. The plane costs one
f32 O(seq²) buffer on one attention at latent resolution; `estimatePeakBytes` reports
it at 4 bytes so the ladder tiles when a whole-image plane will not fit. Measured
after: SD1.5 **1.1e-3** and the Flux VAE both match the CPU decoder.

⚠️ **`attn_out` runs its OWN online softmax.** There is no softmax pass between it and
`attn_scores`; adding one exponentiates twice — finite, plausible, and wrong by rel L2
0.26. `dit_gpu`'s f32 branch omits it for the same reason, and that absence reads as a
missing step until you read the kernel.

**How `zimage_cuda` differs from `zimage_gpu`** — two things, both forced by the op
surface rather than chosen:

- ⚠️ **The modulation table is the FOLDED one** (`zimage.modulationTableFolded`).
  CUDA has no standalone `modulate` — only `rmsMod` (`out = x*inv*premul[col] +
  shift[col]`) — so the pre-norm weight must arrive already multiplied into the scale
  (`premul = norm_w * (1 + scale)`, exact since both are per-column multiplies).
  Vulkan takes the *unfused* table because it has a separate `modulate`. Both come out
  of one AdaLN evaluation so they cannot drift, and each backend is compared against
  the same CPU forward.
- **Attention is `opAttnTC`** (cuDNN's fused SDPA under `--backend cuda`, hand-PTX
  otherwise). The naive `be.attn` is the fallback behind `force_naive_attn`.
- **No new kernel was needed on either backend.** The block maps onto the existing
  vocabulary — the main way this port differs from the SD family's, where GroupNorm,
  the SpatialTransformer and the LIFO skip stack were all new.
- **`tanh` the gates on the host**, inside that same table — one `dim`-wide vector per
  block per step, so it costs nothing and saves a kernel.
- **The fused `qkv` splits into three zero-copy ROW VIEWS**, so it is three GEMMs
  writing straight into contiguous q/k/v buffers. ⚠️ The device weight cache keys on the
  host pointer, so part 0 shares the fused tensor's pointer — never upload both.
- **Head width is exactly 128**, what the coop/PTX attention builders tile, so unlike
  the SD family there is **no head padding**; every trunk GEMM width (3840 / 10240 /
  11520) is a multiple of 128 too.
- **Keep on the host**: the timestep MLP and all AdaLN linears (precomputed for the
  whole schedule at `Session.init`), the entire caption half (⚠️ `context_refiner` is
  `modulation=False`, so it is per-*image*, not per-step), patchify, and the final layer.
- ⚠️ **Two sequence lengths per forward**: the `noise_refiner` blocks run on the image
  half alone at the positions it will hold in the joint sequence, so the rope table is
  *sliced*, not rebuilt; the trunk then runs on the concatenation.
- ⚠️ The Q/K norms take `finfo(f32).eps` (1.19e-7), **not** the blocks' 1e-5.
- ⚠️ `ctx.independent(n)` counts **dispatches, not calls**, and a coop GEMM is three of
  them sharing scratch — do not mark the three qkv GEMMs independent.

### 2A. The SD family (SD1.5 / SDXL)

Every stage of the SD pipeline runs on every backend (`clip_text{,_gpu,_cuda}.zig`,
`sd_unet{,_gpu,_cuda}.zig`, `sd_vae{,_gpu,_cuda}.zig`).

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **CLIP-L / CLIP-G** | ✅ ref | ✅ | ✅ | ✅ | `clip_text{,_gpu,_cuda}.zig` |
| **UNet** (LDM `UNetModel`) | ✅ ref | ✅ | ✅ | ✅ | `sd_unet{,_gpu,_cuda}.zig` |
| **VAE decode** (AutoencoderKL) | ✅ | ✅ | ✅ | ✅ | `sd_vae{,_gpu,_cuda}.zig` |

⚠️ **The CLIP towers were CPU-only until 2026-08-03** ("they run once per render and cost
0.0–0.9 s"), and that stopped being true when long-prompt chunking landed: a real booru-style
prompt is two or three 77-token windows, positive and negative, across two SDXL towers, so the
encode became **1.7 s** measured at 1024×1536 — more than a whole 512² sampling step. The batch
axis on the device is the *chunk*, since a single 77-row window gives only 77·heads threads.

Three kernels were added for it, all CLIP-specific:

| kernel | why it is not an existing one |
|---|---|
| `attn_causal_batched` (vulkan) | CLIP's tower is a causal LM body used as an encoder; every other encoder here (SigLIP, Snowflake, the ViTs) is bidirectional. CUDA needed nothing — `Backend.attn` already takes a `causal` flag. |
| `gelu_quick` (both) | CLIP-**L**'s FFN, `x·σ(1.702x)`. Only the *gated* form existed. |
| `gelu_erf` (both) | CLIP-**G**'s FFN. Only the gated form (`geglu`) existed. |

⚠️ **The two towers of one SDXL checkpoint use DIFFERENT activations**, and the three GELU forms
here agree to ~1e-2 — close enough to look correct, far enough to shift style. `cfg.act` carries
which; the parity tests compare value by value rather than in aggregate for exactly that reason.

⚠️ **Precision differs between the two GPU arms, and it is not the caller's choice on CUDA.**
Vulkan follows `qwen3_gpu`'s convention — f32 GEMMs by default, tensor-core f16 only under
`--encoder-f16` — so it tracks the CPU forward to **1.1e-5** rel L2. CUDA has only a tensor-core
entry point at these widths, so it runs f16 regardless and lands ~1e-3, the same regime
`sd_unet_cuda` already runs the denoiser in. A `cuda` render and a `cpu` render of the same seed
therefore differ slightly at the conditioning, which is worth knowing before attributing such a
difference to something else.

**Measured** (RTX 3090, ReleaseFast, 8 steps, CFG 7.5 so two UNet forwards per step;
`s/step` includes the first step's one-time weight upload amortized over eight):

| arm | cpu | vulkan | zig-cuda | cuda | cuda speedup |
|---|---:|---:|---:|---:|---:|
| SD1.5 512², s/step | 6.91 | 0.92 | 0.46 | **0.27** | 26× |
| SD1.5 512², total | 60.5 | 7.9 | 4.1 | **2.4** | 25× |
| SDXL 1024², s/step | 50.06 | 2.20 | 1.83 | **1.11** | 45× |
| SDXL 1024², total | 430.2 | 20.0 | 16.4 | **10.6** | 41× |

So **25–41× end to end on `cuda`**, and 7.5–21× on vulkan. All three GPU arms agree with the CPU
reference at **31.4–31.8 dB PSNR / SSIM 0.968–0.971** over a paired render, holding
97.5–97.7% of its gradient energy — f16 tensor-core differences amplified by eight
sampling steps, not a different picture. The CPU path stays the reference; none of
these are bit-identical to it.

⚠️ **The SDXL VAE overflows f16, and every GPU arm now divides the residual stream before the
cast** (fixed 2026-08-03). Measured with a per-stage probe at a 64² latent: the decoder's
residual reaches **4.2e5** against f16's ceiling of **65504** — the last upsample convolution's
f32 output is 1.26e5 and the next block's 1×1 shortcut reads it — so the activation cast
produced `inf`, the following GroupNorm spread it to NaN through its mean, and **every SDXL
render at 512² or larger came out a solid white image with no error anywhere** on `vulkan`,
`zig-cuda` and `cuda` alike. SD1.5's VAE peaks near 7e3, two orders lower, which is why the
family the code was written against never showed it; it is also why ComfyUI decodes the SDXL VAE
in fp32 (or ships the "fp16-fix" weights) by default.

The fix divides only the *residual-reading* convolutions (the 1×1 shortcut and the level
upsamples) by **256**, a power of two, so the correction is exact — it shifts the exponent and
f16 keeps all 11 mantissa bits — and free, because both halves ride in kernels that already
touch every element (`f32_to_f16_pad2d`/`f32_to_h16_pad` and `bias_compact`, whose spare `f32`
push-constant this now uses; **every caller must pass 1.0, not 0.0**). `opConvF16Scaled` /
`opMatmulCoopF16WScaled` and `convIntoScaled` are the entry points; `residual_act_div` in both
`sd_vae_cuda` and `sd_vae_gpu` carries the measurement. Verified: the SDXL VAE now matches the
CPU decoder at **rel L2 ≤ 1.4e-3 at every size from 12×10 to 128×128 and at both z magnitudes**
in `sd-cuda-test` (both arms), and a real 768² render decodes at **44.0 dB (cuda) / 44.1 dB
(vulkan) against a CPU decode of the same latent**, 64.6 dB through the tiled path.

**Both kernel sets are pinned against the CPU ops they reproduce**, by different mechanisms:
the SPIR-V ones in `sd_unet_gpu`'s test block (`-Dintegration` + the `testdata/gpu-tests`
marker), the PTX ones by `TensorPencil sd-cuda-test [<ckpt>] [libs]` — a CLI command, because
the test binary brings up no CUDA context, matching `cuda-vae-test` / `cuda-dit-test`. Both
cover GroupNorm (mean 0 and mean 400), the erf-GELU gate, cross-attention at unequal q/kv
lengths, self-attention at all four SD head widths, the convolution at all three sampling
modes, the channel concat, and the head pad/unpad round trip; the CUDA command adds
`add_bias_rows`, the VAE mid-block's own attention shape (one 512-wide head, seq 384/4096/9216),
and whole-stage UNet and VAE parity, and exits non-zero on failure. ⚠️ **The VAE parity check is
a SIZE × MAGNITUDE sweep** (12×10 … 128×128, z at ×1 and ×7.7) and both axes earned their place:
the old single 12×10 unit-gaussian check passed throughout the months the SDXL overflow above
made every real 512²-and-larger render white. `sd-cuda-test` also detects the checkpoint's family,
so an SDXL file exercises the VAE sweep (the two families' decoders are architecturally identical);
the UNet check stays SD1.5-only. Run it in
both arms: `libs` reports "runs at 40/64/80/160" against hand-PTX's "128/128/128/256", which is
the padding difference above made visible.

⚠️ **`zig-cuda` beats `cuda` nowhere here, but `cuda` beats it by less than the DiT's
margin** — SD's GEMMs are small enough that cuBLASLt/cuDNN's per-call overhead eats
into the win, most visibly at SD1.5's 512² sizes.

**Attention head widths are what forced per-backend choices**, and they are the one
place the three GPU arms do not share a shape. SD1.5 attends with head_dim 40/80/160
(fixed 8 heads, so the width grows with the level) and SDXL with 64 (fixed width, so
the head count grows):

| backend | how it attends | cost |
|---|---|---|
| **cuda** | cuDNN fused SDPA at the true width | none |
| **zig-cuda** | zero-padded to a multiple of 128 | `128/hd` on the attention |
| **vulkan** | zero-padded to 128; hd 160 falls back to the scalar kernel | as above |

The padding is exact (a zero dimension contributes nothing to a dot product, and V's
zero columns give output columns that are dropped) — it only costs arithmetic. Both
GPU limits are the same one: the P@V GEMM tiles the head dimension in 128-wide blocks
(`coopmat.buildGemmAttnOut`, `launchHgemmB`'s `grid.x = n/128`), so a narrower head
either launches a zero-sized grid or silently computes part of the head. Removing the
padding means parameterizing those two builders by head width, not changing the model
code. ⚠️ **hd 160 is affordable on Vulkan's scalar fallback only because the wide-head
levels are also the small-`n` ones** — SD1.5's two innermost levels carry 256 and 64
positions at 512².

⚠️ **A CPU VAE decode was the whole pipeline once the UNet moved**: 32 of the 47
seconds of a 1024² SDXL render. On the device it is ~1.2 s, and it agrees with the CPU
decoder at **65.0 dB / SSIM 0.9999** decoding the same latent.

### DiT block weight-dtype support

| DiT block dtype | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| **fp8-e4m3** | ✅ | ✅ (fast coop) | ✅ stream+dequant¹ | ✅ stream+dequant¹ |
| **int8-convrot** | ✅ | ✅ | ✅ | ✅ (+f16 MLP) |
| **int4-convrot** | ✅ | ❌² | ✅ | ✅ |
| **bf16 dense** | ✅ | ✅ native/f16 | ✅ native/f16 | ✅ cuBLASLt R_16BF |
| **f32** | ✅ | ✅ (offload) | — | — |

¹ fp8 block linears stream through `opMatmulFp8` (`backend.zig`): the fp8 weight is decoded to an f16 scratch (`dequant_fp8_f16`, per-tensor scale folded) and run through the validated `buildHgemm` (hand-PTX) or `ltMatmulF16` (cuBLASLt) — the same primitive the fp8 text encoder uses, `LinKind.fp8` in `dit_cuda.zig`. The dequant-to-f16 scratch is re-materialized per GEMM (no fp8 tensor-core GEMM yet), so fp8 on the CUDA backends is correctness-first and slower per step than the native int8 path. The **CUDA fused `opMatmul`** (bias + destination-offset, used only by `first`/`last.linear`) still has no fp8 variant, so those two tiny projections — like bf16 (ComfyUI-native int8 checkpoints) — are materialized to f32 once at load (`DiT.opMatmulF32`, `dit.zig`); otherwise the run either aborts on the fp8 assert or (bf16) reads the packed bytes as f32 → pure noise.
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
| **qwen3** (Qwen3 / Qwen3-VL / llama-arch Mistral-Nemo) | ✅ | ✅¹ | ✅ | ✅ | `qwen3{,_gpu,_cuda}.zig` | only arch with spec decode; primary path fp8 safetensors; GGUF metadata config-detect up to 64 layers (Qwen3-32B); the generalized `llama`/Mistral arch (un-permuted q/k, optional QK-norm, untied head, runtime vocab/eps) shares this stepper; hybrid CPU split like qwen35 (32B: offload measured ~1.7x over the removed streaming path at equal budget) |
| **qwen35** (hybrid DeltaNet: GDN+attn) | ✅ | ✅ | ✅ | ✅ | `qwen35{,_gpu,_cuda}.zig` | GGUF block-quant on vulkan too; no spec (recurrent state) |
| **gemma3** (sandwich norms, dual RoPE) | ✅ | ✅ | ✅ | ✅ | `gemma3{,_gpu,_cuda}.zig` | entirely GGUF block-quant |
| **gemma4** (per-layer geom, factored RoPE) | ✅ | ❌² | ✅ | ✅ | `gemma4{,_cuda}.zig` | `--backend vulkan` rejected; hybrid CPU split like gemma3 (GUI-driven; per-layer-KV host shadow keeps the device ring layout). Config is fully metadata-driven: 12B (Q4_0, 48L) **and 31B** (mixed Q4_K/Q6_K + tied Q6_K head, 60L, hidden 5376, kv 16↔4) both load with no code change; vision via `gemma4uv`/`gemma4v` towers (see §5) |

¹ **qwen3 on vulkan** (`qwen3_gpu.zig VulkanLM`, config-driven) runs both regimes. **Dense** (fp8/bf16/f32, tied head — the Qwen3-VL text encoders): batched square-attention prefill + spec decode; bf16 weights read **natively** (2-byte, `transpose_bf16`/`pipe_tr_bf16` + a bf16 branch in `gemv_partial`/`gemv_partial4`, weight code `context.WCode.bf16`), like CUDA's `gemv_bf16` — no widening, weights stay 8GB; bf16 has no tiled GEMM so prefill streams through grouped GEMV. Generation is coherent (4B ~32 tok/s) but not token-identical to CPU (GEMV reduction-order drift). **GGUF block-quant** (q8_0/q4_k/q5_k/q6_k/iq4_nl layers + untied block-quant head — the `llama`/Mistral arch, e.g. Mistral-Nemo IQ4_NL): **decode** is the per-row fused-dequant GEMV (`gemvW` → `opGemvQuantT`); **prefill of the fresh prompt** runs the tensor-core GEMM — `opMatmulCoopQuant` dequants each weight to f16 k-major on the GPU (`dequant_{fmt}_f32` reading the resident 32-row-group transposed weight → `pack_h16_kmajor`), reusing the existing f16-weight coopmat GEMM (`coopF16WDispatch`), so the whole prompt prefills in one batched pass instead of a forward per token (**~350× faster prefill on a 3090: a 411-token prompt went 39 s → ~0.1 s of marginal prefill**; measured token-identical output). The f16 weight is re-dequanted each prefill into one reused scratch (the f16 form is 4× the block-quant size and won't all fit resident); decode and short follow-up turns stay on the exact GEMV. Falls back to per-token GEMV when the device lacks the f16 coopmat pipeline. QK-norm is skipped (`cfg.qk_norm=false`); the F16 embedding is host-gathered to f32 like the dense path. Validated token-identical to CPU/llama.cpp on the constrained sequence prompt. **Only a block-quant token embedding is still rejected** (`llm_main.zig`) — there is no Vulkan block-quant gather kernel, so it needs an f16/bf16 embed table. **Opt-in int8 dp4a decode** (`TP_VK_DP4A=1`, q8_0/iq4_nl, needs `VK_KHR_shader_integer_dot_product`): repacks the weight into an int8-interleaved layout (`repack_q8_0`/`repack_iq4_nl`; iq4_nl codebook pre-applied) so a warp reads 4 contiguous quants/`u32` and dots via `OpSDot` — **~2.2× faster decode (9.6 → 20.7 tok/s on a 3090)**, output correct. The `OpSDot` capability is injected by `spv.zig withDotProduct` into a separate `dp4a` SPIR-V module (kept off the shared eltwise module). The repacked weight ~doubles the iq4_nl VRAM (Nemo 7 → 13 GB), hence opt-in until VRAM-aware auto-sizing lands; when on it serves both decode and prefill dequant (one resident copy).
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

DType enum: `src/dtype.zig:11` — `f8_e4m3, f16, bf16, f32, i8, i4, q4_0, q8_0, q4_k, q5_k, q6_k, iq4_nl, q1_0, q2_0_g64, q2_0_g128`.
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
| **GGUF iq4_nl** | ✅ ggml | ✅ `gemv_iq4_nl`/`_t` (scalar, module-const LUT) | ✅ `gemv_iq4_nl` (shared-mem LUT), `dequant_iq4_nl_f16` | ⤷ dequant→f16 | 32 elems / 18 B; non-linear `kvalues_iq4nl` LUT |
| **GGUF q1_0** | ✅ ggml | ❌ (no GEMV kernel) | ✅ `gemv_q1_0` (f32), `gemv_q1_0_q8` (dp4a, default), `mmq_pipe_q1_0`, `dequant_q1_0_f16` | ⤷ dequant→f16 | 128 elems / 18 B; **1 sign bit per weight**, `v = bit ? d : -d`, `d = mean\|x\|` |
| **GGUF q2_0 g128** | ✅ **native** `dotQ2_0G128` (fused, exact activations — NOT ggml) | ❌ (no GEMV kernel) | ✅ `gemv_q2_0_g128_q8` (dp4a, default), `gemv_q2_0_g128` (f32), `dequant_q2_0_g128_f16` | ⤷ dequant→f16 | 128 elems / 34 B; 2 bits/weight, `v = (code - 1) * d`, codes → {−1, 0, +1, +2} |
| **GGUF q2_0 g64** | ✅ ggml | ❌ | ✅ `gemv_q2_0_g64_q8` (dp4a), `gemv_q2_0_g64` (f32), `dequant_q2_0_g64_f16` — ⚠️ **built, not measured** (no g64 file exists here) | ⤷ dequant→f16 | 64 elems / 18 B; same arithmetic, ggml's own `QK2_0` |

Notes:
- **int4 / W4A4 is CUDA-hand-PTX-only** (`s4 m16n8k64` tensor cores). CPU has a correctness path; Vulkan and cuBLASLt cannot do sint4.
- **GGUF block-quant on GPU dequants on-the-fly inside the GEMV** — never expanded to VRAM. Vulkan's GEMV is **scalar f32 (no dp4a)** and lacks `q4_0`. The `cuda` (libs) arm dequants GGUF to f16 for prefill GEMM but uses the shared **hand-PTX** GEMV at decode (cuBLASLt/cuDNN never consume GGUF block-quant directly).
- **convrot** (`src/ops/convrot.zig`): size-256 Hadamard rotation, applied at int8/int4 dequant. `cols` must be a multiple of 256 (i4 also even from nibble-packing).
- **q1_0 is the first 128-element block** (every other quant here is 32 or 256), so anything that hardcoded those two sizes is wrong for it — `matmul.packedTaskBlock`'s k-slice assert is now per-dtype for that reason.
- ⚠️ **GGUF type id 42 is claimed by TWO formats, and picking the wrong one is silent.** Both spell
  themselves "Q2_0", both compute `v = (code - 1) * d` from 2-bit codes packed 4 per byte LSB-first, and
  they differ **only in block size**: upstream ggml's `QK2_0 = 64` (18 B) against the PrismML llama.cpp
  fork's `prism` branch at `QK2_0 = 128` (34 B). Every published Bonsai / "Ternary" GGUF is the latter,
  advertised on the model card as "Q2_0 g128"; the fork's own `master` and `pr/q2_0-*` branches are g64,
  so the fork is not a reliable tell either. Nothing *inside* a file distinguishes them — not the type id,
  not `general.file_type` (41 = `LLAMA_FTYPE_MOSTLY_Q2_0` for both).
  - **The only signal is the on-disk row length, and it differs by just 17/18.** On Bonsai's 5120-wide row
    that is 1440 B (g64) vs 1360 B (g128). `gguf.detectQ2_0Variant` resolves it from the gap to the next
    tensor in offset order, accepting a candidate only on `alignForward(size) == gap`; ~5.6% against ≤32 B
    of padding leaves no ambiguity band for a real weight matrix, and a gap fitting both or neither casts
    no vote rather than guessing. No vote at all, or two tensors disagreeing, is `AmbiguousQ2_0Variant`.
  - ⚠️ **The failure is asymmetric, which is why detection is not optional.** Reading a **g64** file as
    g128 computes *smaller* spans than reality, so `gguf.zig`'s bounds check (`offset + nbytes <=
    payload.len`) **passes** and every tensor view is silently short and misaligned — plausible garbage,
    no error. The reverse (g128 read as g64) overruns and does surface as `InvalidOffsets`, which is how
    this was found. A `ne[0] % blockElems()` check does not save you: real hidden dims (5120, 4096, 14336)
    are multiples of 128 as well as 64.
  - **g128 is decoded natively, NOT by ggml** (`quants.dequantQ2_0G128`), even though our pinned ggml has
    a `GGML_TYPE_Q2_0` — because that one is g64, and `QK2_0` is a compile-time `#define` baked into every
    kernel, so there is no way to ask ggml for the other stride. `quants.ggmlType(.q2_0_g128)` returns
    **null** as a hard interlock: it makes every ggml entry point (dequant, `vec_dot`, quantize)
    unreachable for the dtype. Patching the vendored ggml's `QK2_0` was rejected — it would fork a
    dependency and break g64 instead, trading one silent wrong answer for another.
  - **Cost of that interlock, and its fix:** g128 loses ggml's AVX2 fused `vec_dot`, so it first fell back
    to the packed dequant-to-f32 path — 4 B/element against 0.266 B stored, ~15x the memory traffic, which
    measured **0.2 tok/s** decode on Bonsai-27B. `quants.dotQ2_0G128` + `matmul.native_gemv` are the native
    replacement (same thread-over-row-chunks shape as `ggml_gemv`, no activation quantization):
    **0.2 → 2.6 → 3.2 tok/s, 16x**, output byte-identical throughout. g64 keeps ggml and is unaffected.
    - The 2.6 → 3.2 step is the instructive one. The first kernel used a `[256][4]f32` coefficient table
      indexed per qs byte — the obvious form, and **three loads per 4 elements**. `--profile` put `matmul`
      at **91.2%** of decode while sustaining only **17.8 GB/s** of weight traffic, far under this box's
      DDR5 bandwidth: load-issue bound, not memory bound. Replacing the table with SIMD shift-extraction
      (one integer load per 8 codes) gave +23% at **21.9 GB/s**, `matmul` still 90.3%.
    - **vs llama.cpp: 0.76x**, measured matched — same prompt, same 299 generated tokens, `-t 8` on 8
      cores (SMT off), two runs each: **TP 3.3, 3.3** against **llama.cpp 4.31, 4.36** (PrismML `prism`).
      ⚠️ An earlier figure of "0.53–0.68x" in this file was **wrong**: it compared a 40-token TP run against
      llama.cpp at `-t 12`, and short runs here are noisy enough to be meaningless — the same 39-token
      llama.cpp workload measured **6.02 tok/s at `-t 12` and 4.06 at `-t 8`**, i.e. *faster* while
      oversubscribed. Two 300-token runs per side is the smallest thing that settles it.
    - The remaining 1.31x is llama.cpp's kernel design, not a missing micro-optimization. Its x86
      `ggml_vec_dot_q2_0_q8_0` **quantizes the activation to q8_0** and dots in int8: `vpdpbusd` (VNNI —
      present and enabled on this CPU) does 32 multiply-accumulates per instruction against our 8 f32
      FMAs, and the activation stream is 4x smaller. It does not net 4x because it spends more shuffling
      to widen 2-bit codes to bytes, needs a **second** `vpdpbusd` per chunk for the `-1` offset
      (`dot(code,y) - dot(1,y)`, since codes are 0..3 but values are −1..2), and pays a horizontal sum per
      32 elements.
    - **Closing it means quantizing the activation, which this kernel deliberately does not do** — the
      same accuracy trade `exact_activations` exists to name. At 0.76x on a fallback path, against an
      exactness property the packed path shares, that has not been judged worth taking.
- **q2_0 g128 measured on Bonsai-27B** (`Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0`, 7.2 GB at 2.125 bpw),
  3090, `zig-cuda`, 7.2 GB VRAM. Reference is a **PrismML `prism`-branch llama.cpp built with CUDA**
  (`-ngl 99`), same prompt and 299 generated tokens, two runs each:

  | | TP | llama.cpp | ratio |
  |---|---|---|---|
  | decode, f32 GEMV (first cut) | 32.4 | 67.8 / 66.3 | 0.48x |
  | decode, **dp4a** (session A) | 62.8 / 61.7 | 67.8 / 66.3 | 0.93x |
  | decode, **dp4a** (session B, interleaved) | 57.7 / 57.7 / 58.6 | 61.9 / 63.3 / 64.2 | **0.92x** |

  ⚠️ Compare down the *ratio* column, never across the tok/s columns between sessions — see the
  measurement note below.

  **PREFILL** (1937-token prompt, so the ~0.3 s of fixed PTX-JIT that swamps a 44-token prompt is only
  ~7% here). `mmq_pipe_q2_0` closed most of the gap:

  | | TP | llama.cpp | ratio |
  |---|---|---|---|
  | prefill, dequant+f16 | 463 / 472 / 486 | 1206 / 1140 | 0.40x |
  | prefill, **MMQ** | 768 / 817 / 770 | 1194 / 1287 / 1231 | 0.63x |
  | prefill, MMQ + **weight upload out of `pp`** | **922 / 943** | 1178 / 1165 | **0.80x** |
  | same, at N=497 | 943 / 1025 | 1204 / 1052 | **~0.88x** |

  ⚠️ **The last row is an ACCOUNTING FIX, not a speedup — no user-visible time changed.**
  `Backend.cachedWeight` uploads lazily, so ~600 ms of host->device weight copy was landing inside
  the FIRST forward, i.e. inside `pp`. llama.cpp offloads at model-load time and reports it as `load
  time`, outside `prompt eval time`, so every prefill ratio above the last row was measured against a
  competitor that excluded a cost we included. `CudaLM.warmWeights` (called from `session.run` before
  `prefiller.prefill`) now uploads up front; the time simply moved to the `setup` figure, which went
  0.0s -> 0.6s. Evidence it was exactly this: a warm 13-token one-shot prefill went **743 ms -> 166 ms**,
  and three turns piped into one process had always shown 47 ms then 14 ms for the same prompt.
  ⚠️ Prefill tok/s is now **flat in prompt length** (1031/954/956/888 at 497/977/1937/3857) where it
  used to *rise* (520/690/752/822) — that rise was the fixed cost amortizing, and it is the signature
  to look for if this regresses.
  ⚠️ **Only qwen35 has `warmWeights` so far** — it is opt-in via `@hasDecl`, so qwen3/gemma3/gemma4
  still charge the upload to their prefill and their prefill numbers are understated by the same
  bug. Each needs ~30 lines mirroring its own `layerDeviceBytes`.

  Isolated A/B from ONE binary (`TP_NO_MMQ2`), alternating: **486 → 780 tok/s, +61%**, with decode
  unchanged (58–59 both arms) and output still correct — including a 1946-token multi-chunk prefill.

  **Why the port was cheap, and this is the reusable part:** the q1_0 prefill campaign's 333 → 1097 was
  five architecture-level wins (batched GDN ops, register-resident GDN recurrence, group-shared prefill
  attention, per-arch prefill batch 512) plus two q1_0-kernel ones. **All the architecture-level work is
  dtype-independent and was already active for q2_0** — verified: `prefill_batch = prefill_chunk = 512`
  via `prefillBatchOf`, plus `opGdnDeltaChunk` / `gdn_conv_batch` / `buildAttnSplitGroup`, none of which
  look at the weight dtype. So only the one dtype-specific piece was missing.
  - `buildMmqPipeQ2_0` reuses `buildMmqPipeQ1_0`'s pipe wholesale because q2_0 shares **both** properties
    that make q1_0 the simplest MMQ here: signed symbols out of the `prmt` table mean **no min term**, and
    a 128/64-element block means **one scale per 64-k slab**. Only the addressing (16 code bytes per slab
    not 8, stride 34/18, `SPB` slabs per block) and the unpack differ.
  - ⚠️ **The one real risk was registers, and it had to be measured first.** `ptxas -arch=sm_86 -v` puts
    `mmq_pipe_q1_0` at **255 registers with zero spill** — the hard ceiling. A naive port holding eight
    live `u16` instead of four would have spilled. Packing each pair into a u32 *at load time*, into the
    addressing temps that are already dead, keeps it at **255 registers / 0 spill / 44032 B smem —
    identical to q1_0** — and a u32 of 16 codes is exactly what `emitQ2Unpack` wants anyway.
  - ⚠️ `prmt` reads only the **low 16 bits** of its selector, so a u32 of 16 codes is unpacked in two
    halves rather than masked in one go.
  - ⚠️ g64's MMQ is generated from the same function but, like its GEMVs, is **built and never executed**
    — no g64 file exists here.

  **The prefill budget after MMQ** (`TP_SKIP` bitmask in `qwen35_cuda.stepBatch`, one component removed
  per run, 1937-token prompt, each measurement PAIRED with an adjacent baseline so the box's drift
  cancels). The parts sum to the 2443 ms baseline exactly — that closure is what makes it usable:

  | component | ms | share |
  |---|---|---|
  | MLP (gate/up/down + silu) | 910 | 37% |
  | **per-token OTHER** (norms / rope / KV store / elementwise) | **393** | **16%** |
  | **fixed startup** (module load + PTX JIT) | **372** | **15%** |
  | GDN qkv/z/out GEMMs | 307 | 13% |
  | GDN recurrence + batched ops | 172 | 7% |
  | attention op | 158 | 6.5% |
  | attn qkv/o GEMMs | 131 | 5.4% |

  - ⚠️ **A single unpaired sweep was useless** — the baseline itself moved 773→806 between runs and
    `TP_SKIP=4` came out *below* baseline, i.e. removing work appeared to make it slower. Pair every
    skip run with an adjacent baseline.
  - **The fixed term was mostly the lazy weight upload, now moved out of `pp` (see above).** What is
    left of it is module load + PTX JIT + first-touch allocation; `CUDA_CACHE_DISABLE=1` costs +344 ms,
    so JIT is a real per-process cost, and `eltFn` loads one module per distinct PTX string in the
    shared `Backend`. Historical note on how badly this was mis-estimated before the cause was found:
    Four methods disagree badly: a 4-point fit on the baseline gives ~372 ms, refitting per *chunk*
    (cost is stepwise in `ceil(N/512)`, not linear in N) gives ~422 ms, a 3-point fit on the
    `TP_SKIP=511` floor gives 584 ms, a warm 13-token one-shot measures 743 ms — and three turns piped
    into ONE interactive process cost 47 ms then 14 ms, i.e. only ~32 ms of one-time cost. They cannot
    all be right, so what the term is *made of* is still unknown. Candidates not yet separated: CUDA
    module load, PTX JIT, first-touch device allocation, and the lazy host→device weight upload
    (`cachedWeight` uploads on first use, so a one-shot run pays ~7 GB of PCIe *inside* the `pp`
    timer — where llama.cpp uploads at load time, outside its `prompt eval time`).
  - ⚠️ **That last point means the prefill ratios above may be unfair to us**, since our `pp` can
    include a one-time weight upload that their `prompt eval time` excludes. Settle it before quoting
    prefill parity again.
  - The one solid cross-cutting figure: `CUDA_CACHE_DISABLE=1` costs **+344 ms**, so PTX JIT is a real
    per-process cost whenever the CUDA cache misses — and `eltFn` loads one module per distinct PTX
    string, in the shared `Backend`, so every CUDA model pays it.
  - ⚠️ **MMQ is NOT the remaining bottleneck, and this is the receipt that stops another round of
    kernel tuning.** The MLP's three GEMMs are 66.3 TFLOP of int8 work in 910 ms = **72.9 TOPS**,
    against llama.cpp's isolated `test-backend-ops` q2_0 MMQ at **79.56 TFLOPS** — **0.92x, in situ
    against their microbenchmark**, and microbenchmarks flatter (one L2-resident weight). Further MMQ
    work would have to beat their kernel, not catch it.
  - ⚠️ **"Fuse the elementwise layer" was the wrong target and the sub-budget killed it.** Splitting
    the non-GEMM per-token work with four more bits: **norms 120 ms**, adds+gating 19, rope+deinterleave
    12, KV store ~0 — **151 ms, ~6% total**, so perfect fusion of all of it wins ~3%. The "393 ms of
    elementwise" that motivated it was mostly a bad fixed-cost estimate, not real per-token work.
  - **Next levers, in order:** (1) characterize the fixed term — it is the largest non-GEMM item by
    every estimate and nothing should be built on it until its composition is known; (2) norm fusion
    at 120 ms / 4.7%, contained but small; (3) GDN recurrence at 172 ms — the chunked DeltaNet that
    would shrink it **costs bit-identity** with the decode path and was ruled out for q1_0 on exactly
    that ground. MMQ is done (0.92x of llama.cpp's kernel in situ).

  - **The dp4a GEMV is worth ~1.9x** (32.4 → ~58–63), the same step q1_0 took (37.2 → 75.8). Both kernels
    are generated from one `Q2Geom` template, so g64 and g128 cannot drift apart. This is the only change
    here that is far outside the noise floor; everything below is inside it.
  - ⚠️ **MEASUREMENT METHOD, and it invalidated three of this section's earlier conclusions.** This box
    drifts several percent *between sessions*: the identical binary measured 62.8 tok/s in one session and
    57.3–59.4 (5 runs, ±1.8% within-session) in the next, and **llama.cpp moved with it** — 67.8 → 63.1
    over the same interval. Absolute tok/s across sessions is therefore meaningless; only an
    **interleaved, same-session A/B** is. Clocks are not the story (they boost to 1950–1980 MHz of 2100
    either way) and neither is short-run noise — the within-session spread is small. Whatever drifts,
    drifts for both engines, so the RATIO is the stable quantity and it has held at **0.92–0.93**
    throughout.
    - Casualty 1: "the `prmt` symbol-LUT added ~2%" — inside the noise, unproven.
    - Casualty 2: "two-row blocking is 4% slower" — **wrong**, it was compared against a stale baseline
      from the previous session. Re-measured properly (same binary, `TP_Q2_X2`, alternating): 1-row
      56.9 mean vs x2 57.7 mean, i.e. **a wash**. The kernel is kept behind `TP_Q2_X2`, default off.
    - Casualty 3: "the residual is `quantizeX` redundancy" — **wrong**, `step` already hoists it (one
      quantize serves qg/k/v, qkv/z/alpha/beta, gate/up). The ~256 per token that remain are all for
      fresh data; fusing every one of them away bounds out near 2.5%.
  - **Occupancy is already 100%, measured** (`ptxas -arch=sm_86 -v`: 38 registers, 256 threads/block →
    8 warps/block × 6 blocks = 48 warps/SM, the Ampere maximum; full occupancy needs ≤42.6 registers).
    So "more parallelism" is not available and not the lever — which is also why two-row blocking could
    not win: it halves the block count with no occupancy headroom to reclaim, and on a 5120-row weight
    that drops the grid from 640 blocks (1.3 waves over 82 SMs) to 320 (0.65), under one wave.
  - **Where the residual ~8% actually lives is still OPEN, and the next step is a per-op comparison, not
    another guess.** llama.cpp's isolated kernel on `m=4096, n=1, k=14336` is **26.72 us/run** =
    15.6 MB / 26.72 us = **~584 GB/s, 62% of peak** (`test-backend-ops perf -o MUL_MAT -b CUDA0
    -p type_a=q2_0`). Their *end-to-end* decode is only ~430 GB/s-equivalent, so a large part of the gap
    for both engines is everything that is not the GEMV. `qgemv-bench` does not yet cover the m=1 decode
    GEMV; adding it would give our directly comparable number against that 26.72 us and finally localize
    the difference to the kernel or to the surrounding pipeline.
  - One structural fact both engines share: a q2_0 block is 18/34 bytes, so `qs` is only 2-byte aligned
    and both issue four `u16` loads per 32 elements (2 bytes per load instruction). The `+2` f16 scale
    makes alignment alternate with the block index, so it cannot be specialized without warp divergence.
  - **Correctness:** the 220-token content run is exactly right (30 integers, alphabet, countdown), and
    against llama.cpp's CUDA output it is character-identical for ~40 tokens before flipping
    `comma separated`/`comma-separated` and re-converging — the near-tie signature both engines' int8
    activations produce, not a decode error. The f32 `gemv_q2_0_*` kernels remain as the exact path.
- **q1_0 measured on Bonsai-27B** (a qwen35 hybrid, 3.8 GB at 1.125 bpw), 3090, `zig-cuda`, against llama.cpp with full offload. **Token-identical to both the CPU path and llama.cpp on 5 greedy prompts** (including a 1163-token multi-chunk prefill), and the decode number is *better* than llama.cpp's:

  | | TP | llama.cpp | |
  |---|---|---|---|
  | decode | **~73 tok/s** | 58.3 | **1.25×** |
  | prefill (1157 tok) | **~554 tok/s** | 1276 | 0.43× |

  ⚠️ **Measure these warm and on an idle GPU.** The 3090 sits at ~1100 MHz of a 2100 MHz max when it has been idle, boosts to ~1935 MHz under sustained load, and a competing process moves these numbers ±8% — enough to invent a regression that is not there. Decode sampled 72–77 and the 9B q6_k prefill 470–521 across this work; a single short run cannot support a few-percent claim.

  Decode came in three measured steps, and the first is the generalizable one: **block-per-row 28.0 → warp-per-row 37.2 → dp4a 75.8**. q1_0 is the format most punished by a block-per-row shared-memory reduction, because one bit per weight means the least work per row of any quant — the reduction costs more than the dot product.
- **Prefill: 187 → 333 → 414 → 554 tok/s**, in that order (2.96× overall). `mmq_pipe_q1_0` (see `buildMmqPipeQ1_0`) plus `prefill_chunk` 128 → 256 gave the first step; isolated, since both changed at once: 187 (chunk 128, dequant) → 251 (chunk 256, dequant) → 288 (chunk 128, MMQ) → **333** (both), so the two contribute ~independently. 512 measures flat against 256, so 256 is the knee. Batching the GDN recurrence (next bullet) gave the second step, to **414**.
- ⚠️ **CORRECTION to an earlier claim in this file: the remaining prefill gap was NOT MMQ kernel quality**, and the "next levers" recorded here (`cp.async`, `ldmatrix`, wider n-tile) were the wrong target. Measured with `zig build qgemv-bench` on the real shape (17408×5120, n=256): `mmq_pipe_q1_0` does **63.9 TOPS**, i.e. **22.5% of a 3090's ~284 dense int8 peak — the same efficiency band as llama.cpp's MMQ** (~24%). Its time is also *flat* from n=1 to n=128 (0.372 ms), the signature of latency, not throughput: at BM=128 with 128 threads the grid is only `rows/128` = 136 blocks and register pressure (128 f32 accumulators) caps occupancy near 2 blocks/SM. The FFN GEMMs account for only ~26% of prefill, so even doubling MMQ would buy ~13% end to end.
- ⚠️ **What prefill actually spent its time on: the GDN recurrence ran PER TOKEN inside batched prefill** — 7 launches per token per linear layer (`quantizeX`, two `rows=48` GEMVs, gates, conv, L2-norm, delta step). On Bonsai that is 48 layers × 1157 tokens × 7 = **389k launches**, against 403k counted by the profiler. The isolation ladder, removing exactly that loop:

  | per-token GDN loop | prefill |
  |---|---|
  | all 7 ops | 333 tok/s |
  | minus quantizeX + 2 GEMVs + gates | 372 |
  | only `opGdnDeltaStep` | 411 |
  | loop removed entirely | 614 |

  So the loop was **46% of prefill** and only `opGdnDeltaStep` is genuinely sequential. Six of the seven are now batched over the chunk (`opGdnConvBatch`, `opGdnGatesBatch`, `opGemvQuantQ8Batch`, `opL2NormRowsGrouped`) for **333 → 414 tok/s (+24%)**, and the seventh — the recurrence itself — became `opGdnDeltaChunk`, taking it to **554** (91% of the 614 ceiling), still token-identical to llama.cpp on 5 prompts including a 1163-token multi-chunk prefill.
- **The recurrence did NOT need a chunked/parallel DeltaNet.** ⚠️ Worth reading before anyone reaches for one: `kernels.buildGdnDeltaChunk` keeps the state in **registers** across the whole chunk (one block per v-head, thread `j` owning column `j` of `[d][d]`), which is the *same recurrence in the same order* — so it is **bit-identical** to the per-token `gdn_delta_step`. That was the deciding property, not the speed:
  - The per-token form streamed the state through global memory every token: `[heads][d][d]` f32 read **and** written, ~3.1 MB each way per layer per token, **~350 GB over a 1157-token prefill**. Its own doc comment already said it was "load-latency bound". Registers make that traffic vanish; only the staged k/q and the small per-token gates/v/o touch memory.
  - **Decode can only ever run the per-token form** (n=1), so prefill and decode must agree exactly or a re-prefill (regenerate, variant rollback, suspend/resume) would diverge from the incremental path. A chunked DeltaNet — matmuls plus a 64×64 triangular solve, which is what llama.cpp's `build_delta_net_chunking` does via `ggml_solve_tri` — reassociates the recurrence and would **not** have this property, on top of making chunk size a numerical-stability knob (upstream drops to CS=16 for the KDA variant).
  - **Verified by A/B on the same backend**, not by argument: batched vs per-token over a 1164-token prefill plus 250 greedy tokens produced **identical bytes** (and 310.8 → 579.4 tok/s on that prompt). Any bit of state divergence would almost certainly flip a token within 250.
  - ⚠️ The `i` walk is **fully unrolled** because PTX cannot index a register file — that is what forces this kernel to be *generated* rather than written, and why `d` is baked in and the module is cached keyed by it (`Backend.gdn_chunk_d`). Cost is ~`d` live f32 registers per thread (128 for qwen3.5); it adds no parallelism (still `heads` blocks), so the ~614 ceiling stands.
  - **The conv is a CONVOLUTION, not a recurrence** — `out[t]` depends only on `x[t-3..t]`, so given the state carried *in*, every token is independent. That is what makes it batchable; only the 3-column state roll is sequential, and it moves to a separate one-per-chunk launch because every token's threads read the same incoming state.
  - ⚠️ **`gdn_conv_batch` sums its taps in order 3, 0, 1, 2**, not ascending, to reproduce `gdn_conv_step`'s order exactly. f32 addition is not associative and this model is validated token-identical, so ascending order would make batched prefill disagree with single-token decode in the last bits.
  - ⚠️ **`ssm_alpha`/`ssm_beta` are [48, hidden], which no GEMM here can take**: `launchHgemm` computes `grid.x = rows/128`, so rows=48 is a zero-sized grid (`CUDA_ERROR_INVALID_VALUE`). Hence `opGemvQuantQ8Batch` — the dp4a GEMV with a token dimension on grid.y — rather than routing them through `gemm`.
  - **What remains is NOT in the GDN path.** With the whole per-token loop removed the ceiling was 614 tok/s, and `opGdnDeltaChunk` reaches 554 — so ~90% of the GDN cost is gone and a chunked DeltaNet could buy at most ~11% more while giving up bit-identity.

### Where the remaining ~2× of qwen35 prefill is (measured 2026-08-05)

Isolated with a temporary `TP_SKIP` bitmask in `stepBatch` (one build, one component removed per run) at 1155 tokens. **The budget closes to 0 ms against the 1978 ms baseline**, which is what makes it trustworthy:

| component | ms | share |
|---|---|---|
| MLP (gate/up/down + silu) | 803 | 41% |
| GDN qkv/z/out GEMMs | 273 | 14% |
| attention op | 217 | 11% |
| GDN recurrence + batched ops | 118 | 6% |
| attn qkv/o GEMMs | 113 | 6% |
| residual | 454 | 23% |

⚠️ **The residual is NOT per-token work — it is ~190 ms of FIXED startup** (PTX JIT of the eltwise/MMQ modules) that the reported `pp` rate includes. Solved from two prompt lengths (1155 and 2295 tokens): baseline is `187 ms + 1.578 ms/token`, so **steady-state prefill is 634 tok/s, not the 554–584 a short prompt reports**. Layer elementwise is only 13 ms and the host `embedTokens` is free (577.4 vs 578.2 tok/s with it removed) — both were candidate explanations and both are wrong. **Compare against llama-bench's warm, averaged number using the marginal rate, or a short-prompt `pp` will understate TP by ~10%.**

Against llama.cpp's 1276 tok/s (0.784 ms/token), the gap splits in two:

| | TP ms/token | llama.cpp | gap |
|---|---|---|---|
| matmul | 1.029 (65%) | 0.622 | **1.66×** |
| non-matmul | 0.548 | 0.162 | **3.39×** |

- ⚠️ **Our MMQ really was slower than llama.cpp's, and that was measured, not inferred.** `test-backend-ops perf -o MUL_MAT -p type_a=q1_0 -b CUDA0` reports **86.50 TFLOPS** at m=4096/n=512/k=14336; `qgemv-bench` on that *identical* shape gave our `mmq_pipe_q1_0` **65.0 TOPS** — **1.33×**. (The earlier claim in this file that the two were in the same band came from dividing llama.cpp's end-to-end rate by the compute bound, which silently assumed their prefill was all matmul. It is 79% matmul, not 100%. **Never infer a kernel's efficiency from an end-to-end rate when a per-op benchmark exists.**)

#### Closing most of it: cp.async double-buffering (2026-08-05)

Two probes decided the design, and the first one saved a lot of wasted work:

| probe (wrong output, timing only) | ggml shape, n=512 |
|---|---|
| baseline | 65.0 TOPS |
| scale fold cut 4× (fold 1 of 4 accumulator quads) | 71.5 (**+10% only**) |
| A/B staging removed entirely | 80.4 (**+24%**) |

So the **fold is not the limiter** — which killed a planned 128-element activation requantization that would have cut the fold 4× for ~10%, and would have changed prefill numerics for it. The limiter was **exposed global-load latency**: the single-buffered loop staged, `bar.sync`d, computed, `bar.sync`d, so nothing overlapped the ~500-cycle loads.

Result, `rel` unchanged at 0.00534 throughout (and still token-identical to llama.cpp on 5 prompts):

| shape, n | before | after |
|---|---|---|
| b27 mlp gate/up, n=256 | 63.9 | **72.6** |
| b27 mlp down, n=256 | 56.5 | **80.8** |
| ggml perf shape, n=512 | 65.0 | **76.2** (vs llama.cpp 86.50 → 1.14×, was 1.33×) |

**End to end: steady-state prefill 634 → 698 tok/s.** Decode untouched (76.7; it never uses MMQ).

Three things the implementation commits to:

- ⚠️ **cp.async PAYS FOR ITSELF IN REGISTERS BEFORE HIDING ANY LATENCY.** The kernel sat at 254 of 255 registers, so there was no headroom for prefetch state at all — but moving B global→shared with `cp.async.ca.shared.global` deletes its 16 staging registers, which funds the ~10 the A prefetch needs. Final: 255 registers, 35.8 KB smem (two buffers), **no spills** (`ptxas -v`).
- **A cannot use cp.async** — its bytes are transformed by the `prmt` decode — so A's raw u16s are prefetched into REGISTERS a slab ahead and decoded into shared after the compute that hides their latency.
- ⚠️ **The slab loop is unrolled 2× so buffer parity is COMPILE-TIME**, keeping every shared offset an immediate (`buf * SH_HALF`) instead of spending a register and an add per access. `cols % 256 == 0` makes `cols/kstep` a multiple of 4, so nslab is always even. The prefetch index is **clamped** to the last slab rather than branched on — the extra fetch is garbage that is never consumed, and clamping keeps the address in bounds. One `bar.sync` per slab instead of two.
- **A smaller free win found on the way:** B's staging issued load/store/load/store, so the second pair of global loads waited on the first pair's registers, serializing two latencies that should overlap. Issuing all four `ld.global.v4.u32` before any store was +3%.

⚠️ **`prefill_chunk` still cannot be raised, and it was re-tested after the speedup.** The kernel is much better at n=512 (gate/up 72.6 → 88.9, down 80.8 → 92.2), so the balance should have shifted — but 256 and 512 measure **668.8 vs 668.7 tok/s** over three runs each. The attention masking cost (below) grows with chunk size and still cancels the matmul gain exactly; 512 also costs 162 MiB more scratch. **Attention is now the binding constraint on chunk size, so fixing it unlocks the kernel's large-n regime as well as its own 11%.**
- ⚠️ **`qgemv-bench` OVERSTATES the kernel** because it hammers one 11 MB weight for 40 iterations, so the weight is L2-resident; the real model streams 3.6 GB of distinct weights per chunk. Use it for A/B between kernels, not as an absolute.
- ⚠️ **`prefill_chunk` above 256 IS DEAD CODE, and that is why raising it never helped.** The engine slices prefill into `engine.prefill_gate_chunk = 256`-token pieces so the pause/cancel gate can land mid-prefill, so `qwen35_cuda.prefill` never receives more than 256 ids (255 in practice — the last prompt token is held back for the first decode step) and its internal chunk loop never iterates. Raising the constant changes only the activation buffer sizes.
  - **How it was found:** the profile showed *identical launch counts* (9434 / 1216 / 19076) at chunk 256 and 512, which cannot happen if the batching changed. Logging the actual `n` gave `n=255` nine times in both configs. Two earlier hypotheses — masked attention work, then attention cost in general — were both wrong, and both were about the *wrong variable*.
  - **Raising it alone is actively harmful** (−31% on the matmul bucket at 512): the same 255-token work with `mlp_gate`/`mlp_up` doubled to 35.7 MB each and `attn_scratch` to 102 MB, half-used, so it is pure locality loss for zero benefit. **⚠️ Change `engine.prefill_gate_chunk` and `prefill_chunk` together or not at all.**
  - **Fixed by making the batch the STEPPER's choice** (`engine.prefillBatchOf`): a stepper that knows what its kernels want declares `prefill_batch`, everyone else keeps `prefill_gate_chunk` and today's behaviour. qwen35_cuda ties `prefill_batch = prefill_chunk` so the two cannot drift again. **Prefill 959 → 1097 tok/s steady-state (+14%).** Measured at 1024 as well: ~604 tok/s, i.e. the buffers outgrow useful locality, so 512 is the knee.
  - ⚠️ **Pause and cancel are now polled at DIFFERENT granularities, deliberately.** `ops/pause.zig` states that parking mid-kernel would strand half-computed state and held locks, so **pause** still parks only at batch boundaries — a paused worker waits at most one forward, which is the accepted cost of a larger batch. **Cancel** discards its work by definition, so it is safe to unwind mid-forward where parking is not: `engine.publishCancel` publishes the engine's flag to `ops.cancel`'s threadlocal token (which the LLM path never set before), and `qwen35_cuda.stepBatch` polls it **between layers**, returning `error.Canceled`. The existing `errdefer` aborts the recording batch and `self.len` only advances after `endBatch`, so an unwind commits nothing and `cached()` stays truthful. Net: cancel got *finer* (≈one layer, was one 256-token batch) while the batch doubled.
  - ⚠️ **There were FIVE copies of this prefill loop** — three in `engine.zig` (plain / argmax / select) and two in `spec.zig` — and the first fix only patched one, which is why the measurement did not move. That duplication is what let the original cap hide. All five now share `prefillBatchOf` / `publishCancel` / `isCancelUnwind`, and those three have unit tests (a stepper with and without the decl, token publish/restore, and that only `error.Canceled` counts as an unwind).
  - `isCancelUnwind` matches by error NAME on purpose: only steppers that poll the token can produce `error.Canceled`, and an explicit `switch` arm fails to typecheck against the error sets of those that cannot.

#### Prefill attention: share the KV fragment across the query-head group (2026-08-05)

⚠️ **CORRECTION to a claim made twice above and acted on once: `attn_split` is ALREADY exactly causal.** Each `(query, head, split)` warp uses `kv_len = kv_len0 + t`, so no masked work is ever computed and discarded, and there is nothing for a "causal-aware" rewrite to skip. The waste is elsewhere: **each of the `group = heads/kv_heads` query heads re-reads the same KV head.** With 24 heads over 4 kv heads and 256 queries that is a 1536× re-read of K and V.

The model matches the measurement, which is what makes it actionable: **6.29 GB of L2→SM traffic per attention call, 503 GB over a 1155-token prefill, ~252 ms at the 3090's ~2 TB/s L2** — against 217 ms measured for the whole `attn` bucket. Warp-instruction bound was ~27 ms, i.e. 9× of headroom to spend on arithmetic in exchange for traffic.

`kernels.buildAttnSplitGroup` (entry `attn_split_g`) makes one warp own a `(query, **kv head**, split)` and loops the group's q heads inside, so K and V are read **once** for the group. Traffic drops by `group` (6× here); the instruction count per warp rises by ~`group` while the warp count falls by `group`, so total instructions are unchanged.

| | 1155 tokens | 2295 tokens |
|---|---|---|
| before | 660 tok/s | 672 |
| after | **716** | **810** |

**Steady-state prefill 698 → 901 tok/s.** The gain grows with context because attention cost grows with `kv_len` while the matmuls do not.

- ⚠️ **Bit-identical to `attn_split` per head** — same j order, same dot order, same butterfly, same `ex2.approx` softmax sequence — and **verified by A/B on the same backend**, not argued: group vs general over a 1164-token prefill + 250 greedy tokens produced **identical bytes**. This matters more here than anywhere else, because prefill attention writes the KV cache that decode then reads.
- ⚠️ **I built it for hd=128 first and it did nothing, because qwen3.5's `key_length` is 256.** `opAttnDecode` branches `hd == 256 or hd == 512` *before* the hd=128 path, so the kernel was never reached. **Check `head_dim` from the checkpoint before choosing an attention variant** — the builder now takes `hd` and the dispatch is tested first.
- Narrow by design (`Backend.attnSplitGroupOk`): hd ∈ {128, 256}, full causal (`window == 0`, `ring == 0`, no bidir), `seq_q > 1`, `heads > kv_heads`. Everything else keeps the general kernel — and since the two are bit-identical per head, that predicate is a pure performance choice, never a correctness one. The module is cached keyed by `(hd*32 + group)*4 + kv_fmt`, because both the group and the KV decode are unrolled into the kernel.
- **All three KV formats have a variant** (`attn_split_g` / `_f16` / `_q8`). `kernels.emitKvFrag` mirrors `elt.emitKVLoad` instruction for instruction — same load widths, same sign-extend shifts, same multiply order — which is what keeps each variant bit-identical to the `attn_split*` it replaces. ⚠️ The q8_0 arm relies on a lane's fragment never straddling a 34-byte block: rows are block-aligned and the fragment is `dims`-aligned with `dims | 32`, so `elem % 32 == (lane*dims) % 32` and the quant offset `(lane*dims & 31) + 2` is loop-invariant.

  | KV format | general | group-shared | |
  |---|---|---|---|
  | f32 | 687.9 tok/s | **753.0** | +9.5% |
  | f16 | 731.2 | **750.2** | +2.6% |
  | q8_0 | 731.2 | **749.2** | +2.5% |

  **f16/q8 gain far less, and the baselines say why:** their KV elements are 2 and ~1.06 bytes instead of 4, so the general kernel's redundant re-reads already cost proportionally less (note their *baseline* is 43 tok/s ahead of f32's). The group kernel brings all three to ~750, i.e. attention stops being traffic-bound in every format. All three verified **bit-identical** by A/B on the same backend over a 1164-token prefill + 200 greedy tokens.
- gemma3/gemma4 are unaffected: their local layers set `ring != 0` and their global layers are hd=512, so both fall outside the gate. Smoke-tested gemma4-12B after the dispatch change.

#### The MMQ fragment loads had a 4-way bank conflict; padding fixed it (2026-08-05)

⚠️ **This was found while scoping `ldmatrix`, and it turned out to be what `ldmatrix` would mostly have bought.** The A/B fragment load has lane `L` read row `base + L/4` at word `ks*8 + L%4`. With the row stride equal to `kstep` = 64 bytes = 16 words, the row term is `16*gid mod 32` ∈ {0,16}, so **the 32 lanes touch only 8 of the 32 banks — a 4-way conflict on every one of the 32 fragment loads per substep.**

Padding the shared row stride to **80 bytes (20 words)** makes it conflict-free: the row term becomes `20*gid mod 32` = {0,20,8,28,16,4,24,12}, eight distinct multiples of 4, so adding `tf` 0..3 covers all 32 banks exactly once. Verified by enumeration, not assumed.

Padding rather than an XOR swizzle, deliberately: at kstep=64 a row is only 16 words, so the pipe's usual `(row&7)<<4` swizzle crosses into the next row, and the `(row&3)<<4` mask that *does* fit still leaves 2-way conflicts because gid 4..7 alias gid 0..3. Costs 4 KB more shared (44 KB for two buffers, still under the 48 KB default limit).

| shape, n | before pipeline | + cp.async | + padding | llama.cpp |
|---|---|---|---|---|
| b27 mlp gate/up, n=256 | 63.9 | 72.6 | **82.8** | |
| b27 mlp down, n=256 | 56.5 | 80.8 | **84.9** | |
| ggml perf shape, n=512 | 65.0 | 76.2 | **88.3** | 86.50 |

**Our MMQ is now slightly ahead of llama.cpp's at the identical shape (88.3 vs 86.50).** `rel` is unchanged at 0.00534 — padding moves addresses, not arithmetic. **Prefill 901 → 959 tok/s.**

⚠️ **`ldmatrix` itself was then NOT done, and here is the receipt rather than a shrug.** Its value was two things: fewer instructions, and conflict-free access. The second is now captured by the padding. The first is 20 of ~464 instructions per substep (32 fragment loads → 12), i.e. **4.3%** — and the fold probe above showed the instruction stream is not the limiter (cutting 384 of 464 instructions bought only 10%), so 4.3% is worth ~1%. Against that: on sm_86 `ldmatrix` is `.b16` only, which is an awkward fit for an int8 `m16n8k32` fragment and would require re-deriving the shared layout in a kernel that is currently validated bit-identical. **Wrong trade; not attempted.**

**Status:** prefill **333 → 1097 tok/s** across this session (**3.29×**), now **0.86×** of llama.cpp's 1276 (from 0.26×), with the GEMM kernel itself slightly ahead of theirs. Decode 75.8 (1.30× theirs). **Status: prefill 333 → 1097 tok/s across this session (3.29×), now 0.86× of llama.cpp's 1276** (from 0.26×), with the GEMM kernel itself slightly ahead of theirs. Decode 75.8 (1.30×). The `prefill_chunk` question is answered in the bullet above — it was capped by `engine.prefill_gate_chunk` all along.
- **q1_0's MMQ is the simplest of the three** and worth knowing why, because it is the one place the format helps: there is **no min term** (`v = ±d` is symmetric with no zero point, so the activation's per-block quant *sums* — a whole shared-memory region in q4_k — are not needed at all), and **the scale is constant across a 64-k slab**, so the row scale loads once per slab where q4_k reloads a pair per 32-k substep and q6_k per 16. Shared memory is 17.9 KB against q4_k's 20.5 KB.
- ⚠️ **`prefill_chunk` and the GDN kernels are shared by every qwen35 model**, so both changes were checked against a non-q1_0 one: the 9B q6_k is 83.4 tok/s decode (was 82.2) and ~505 tok/s prefill across repeats (run-to-run band 493–521 — a single sample is not enough to call a few percent here). `Bufs` derives its padded row count from `prefill_chunk` via `pad_rows` instead of a second hardcoded 128, so the two cannot drift apart.
  - ⚠️ **`l2norm_rows_g` is a SEPARATE kernel, not a mode flag on `l2norm_rows`**, and that is a measurement, not taste: the plain form is launched once per token per GDN layer on models that don't take the batched path (55k times in a 9B prefill), and folding a runtime branch into it cost ~4% there. Duplicating ~20 lines of reduction is the cheaper trade.
- **The dp4a q1_0 GEMV unpacks sign bits with two `prmt.b32` used as nibble LUTs** (technique from PrismML's llama.cpp fork). ⚠️ The selector must be masked to `0x33333333`: `prmt` reads bit 3 of each nibble as a *sign-replicate* flag, and every source byte in the first LUT has its msb clear, so an unmasked nibble ≥ 8 yields 0x00 and silently forces two weights to −1. Symptom was a model at full speed emitting one token forever.

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
`gn_stats` · `gn_combine` · `gn_apply` · `silu` · `geglu` · `concat_ch` · `attn_cross` ·
`head_pad_h16` · `head_unpad` · `im2col_sd` ·
`attn_causal_batched` · `gelu_quick` · `gelu_erf` (CLIP text tower) ·
`gemv_q8_0{,_t}` · `gemv_q4_k{,_t}` · `gemv_q5_k{,_t}` · `gemv_q6_k{,_t}` · `gdn_gates` ·
`gdn_conv_step` · `gdn_delta_step`

**Vulkan GEMM entry points** (`context.zig`): `opMatmul` (f32/fp8) · `opGemv{,Partial,Quant,QuantT}` ·
`opMatmulCoop{,H16}` (fp8→f16) · `opMatmulCoopF16W{,b,h,Dev}` (f32/bf16/f16→f16; `Dev` takes a
device-resident bias, for the SD UNet's per-forward folded ResBlock bias) · `opMatmulCoopBf16` (native bf16) ·
`opMatmulCoopI8{,Fused}` / `opMatmulI8` / `opI8Gemm` (int8 s8→s32) · `opAttnScores{,Vae}` / `opFlash` / `opAttnOut`.
Coopmat SPIR-V builders in `src/gpu/coopmat.zig` (`buildGemmShared` f16/bf16, `buildGemmI8` int8; **no s4**).
Scores shaders are compiled per head width — 128 (DiT, head-padded SD UNet), 384 (Wan VAE
mid-block) and 512 (SD VAE mid-block) — since `buildGemmScores` unrolls the k-depth:
`opAttnScores` / `opAttnScoresVae` / `opAttnScoresSd`.

### zig-cuda — hand-PTX (`src/gpu/cuda/kernels.zig` GEMM, `elt.zig` elementwise/attn)
GEMM builders: `buildHgemm` (f16/bf16 mma m16n8k16) · `buildIgemmSmem`/`buildIgemmPipe` (int8 m16n8k32, int4 m16n8k64) · `buildPrep` (quant/rotate). **All DiT GEMM formats** use warp-cooperative `ldmatrix.x4`/`.x2` frag loads + an XOR-swizzled (`off^=(row&7)<<4`) conflict-free shared layout (`use_ldmatrix` flag on `buildIgemmPipe` — int8/int4 — AND `buildHgemm` — f16/bf16 dense + attention; on at every runtime site; pure permutation → bit-exact, ~+1–3% on qkv/attn-proj shapes, flat on BW-bound MLP shapes).
GEMV: `gemv_{fp8,bf16,f16,q8_0,q4_0,q4_k,q5_k,q6_k}` + `_q8`/grouped-N `_q8n` dp4a variants.
Attn: `attn`, `attn_split`/`_merge`/`_h256`/`_h512`/`_tree`. GDN: `gdn_{conv_step,gates,delta_step}`.
Plus `im2col`, dtype-pad converts, `dequant_*_f16`, rope/norm/act kernels.
SD family: `gn_stats`/`gn_combine`/`gn_apply` (GroupNorm, Welford statistics) ·
`geglu` (erf-GELU gate) · `concat_ch` · `attn_cross` (seq_kv ≠ seq_q) ·
`im2col_sd` (stride 2 / fused 2× upsample) · `head_pad_h16`/`head_unpad` ·
`add_bias_rows` · `gelu_quick`/`gelu_erf` (the CLIP towers' two FFN activations,
ungated — `attn`'s existing `causal` flag covered the rest).
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
