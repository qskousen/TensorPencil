# Video plan: MiniMax H3

The engine is still image-only: every latent is 4D, `wan_vae.zig` collapses the temporal
axis to T=1, `taehv.zig` takes the first-frame path, and there is no audio anywhere. This
is the plan for the fifth dimension and the second output modality, with MiniMax H3 as the
first family.

Read `comfy/ldm/minimax/{model,vae,audio_vae}.py`, `comfy/text_encoders/minimax.py` and
`comfy_extras/nodes_minimax_h3.py` in the ComfyUI checkout before touching any of this.
Those files are the reference; the section "Silent wrong answers" below is what reading
them bought.

## What H3 is

One flow-matching DiT denoising **video and audio jointly** in a single packed token
sequence. Not a video model with audio attached: the audio rows are in the same attention
pass as the video rows, and the two streams run on different sigma schedules.

```
[text | ref/keyframe cond rows | audio | video]
```

Target streams are always the last two segments, audio before video.

### Shapes, read off the local checkpoint

`h3/10erosMaxInt8Ref2va_v10Beta.safetensors`, 21 GB, `bake: comfyui_TensorWiseINT8Layout_convrot_gs256`:

| field | value | derived from |
|---|---|---|
| `hidden_size` | 5376 | `video_patch_proj.weight [5376, 96]` |
| `num_layers` | 50 | `blocks.0..49` |
| `token_refiner_num_layers` | 2 | `token_refiner.blocks.0..1` |
| `num_attention_heads` | 56 | `qkv_proj [21504, 5376] / (3 * 128)` |
| `attention_head_dim` | 128 | `attn.q_norm.weight [128]` |
| `ffn_hidden_size` | 14336 | `mlp.fc1 [28672, 5376] / 2` (swiglu) |
| `latents_dim` | 24 | `final_layer.video_out [96, 5376] / 4` (patch 1x2x2) |
| `audio_latents_dim` | 32 | `final_layer.audio_out [32, 5376]` |
| `text_dim` | 5120 | `condition_proj [5376, 5120]` |
| `adaln_curve_grid`, `time_embed_dim` | 1025, 8 | `adaln_t_table [1025, 8]` |
| `rope_inv_freq_len` | 16 | `rope.inv_freq [16]` |

This is a **curve-form** checkpoint: there is no `time_embedder`. The adaLN linears
consume interpolated coordinates of a precomputed time-embedding curve, which is why
`time_embed_dim` is 8 rather than a few thousand.

Weight storage is mixed **by design**, not by accident:

- int8 convrot gs256: every block GEMM (`qkv_proj`, `out_proj`, `mlp.fc1`, `mlp.fc2`)
- bf16: norms, `condition_proj`, the whole `token_refiner`
- f16: `adaln_proj.linear`
- f32: `video_patch_proj`, `audio_patch_proj`, `final_layer.video_out`, `final_layer.audio_out`, `adaln_t_table`, `rope.inv_freq`

### The other three components

| component | file | shape notes |
|---|---|---|
| text encoder | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`, 27 GB | Qwen3-VL-32B truncated to 50 layers, hidden 5120, 64 q / 8 kv heads x 128, dense MLP 25600, int8 convrot. Plus the Qwen3-VL ViT (27 blocks, 1152, 3 deepstack mergers) |
| video VAE | `minimax_h3_video_vae_fp16.safetensors`, 5.2 GB | 3D causal CNN encoder (ch 128, mult 1/2/2/4/4/8, space_down 16, time_down 4) + **ViT3D decoder**: 36 blocks, 32 heads x 64 = 2048, gated FFN 8192, patch 16 x 16 x 4, rope theta 100, rope_dim_ratio 0.75 |
| audio VAE | `minimax_h3_audio_vae_fp32.safetensors`, 0.6 GB | 32 kHz stereo, 32 latent channels, 800 samples per latent frame (40 fps). DAC encoder (rates 2/4/4/5/5, latent 2048) + **BigVGAN** decoder (upsample 5/5/2/2/2/2/2, initial ch 1024, resblock kernels 3/7/11, dilations 1/3/5) |

### Frame arithmetic

Frame counts snap to `n % 17 == 5` at 24 fps. Everything else follows:

```
latent_t = 2 if frames <= 5 else ((frames - 5) / 17) * 5 + 2
audio_t  = round(frames / 24 * 40)
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)      # video token k spans FRAME_PER_TOKEN[k % 5] frames
FRAME_RESCALE   = 5 / 3                 # pixel frames -> shared time axis
```

The default 1344x768 / 124 frames gives `latent_t = 37`, a 24 x 42 patch grid, so
**37 * 1008 = 37296 video rows** plus 414 audio rows plus text: a packed sequence of
~38k tokens with full attention. Attention alone is ~41 TFLOP per block, ~2 PFLOP per
step. Every per-step cost figure in this file is an estimate from a FLOP count until
something measures it.

## Decisions already taken

- **Output is a real MP4**, muxed in-process via libavformat/libavcodec. It lives in
  `lib/video/av_helper.c` + `src/av.zig`, a sibling of the existing `lib/vips` shim, and
  links into the **executables only**. The `TensorPencil` library module stays pure Zig,
  so the engine hands back frames and audio samples as plain data and the driver muxes.
- **The A1111 `parameters` block goes into an MP4 metadata tag**, the same way
  `image.zig` puts it in a PNG text chunk. A reader re-renders from that block, and
  DiffKeep already reads format-level tags (`diffkeep_video_read_tag`), so there is a
  consumer.
- **Target is full ref2va**, matching the local checkpoint. Build order still runs
  through t2va, because refs cannot be validated until the trunk is correct.
- **LoRA is a runtime low-rank sidecar in bf16**, not a merge-and-rebake. The base stays
  int8; `y = W_int8 x + (alpha/r) B (A x)`. There is no requantization pass, no
  round-trip loss, and strength is adjustable at runtime. It also generalizes:
  merge-and-rebake only works when the exact bake is reproducible. The hazard is the one
  CLAUDE.md already names for weight storage, a sidecar must gate **every** GEMM call
  site or it silently does not apply on a fast path.

  The local turbo LoRA is **rank 128** (384 for the fused qkv), so the sidecar is a few
  percent of a step rather than the under-1% a rank-32 one would cost. See stage 8 below
  for what shipped and what it measures at.

  Three structural facts, all read out of that file's metadata and all silent when
  wrong:
  - `qkv_fusion: block diagonal B; concat A; alpha multiplied by 3`. The fused factor is
    `A [384, 5376]` (three rank-128 blocks concatenated) and `B [21504, 384]`
    **block-diagonal**. Treating B as dense is numerically right and wastes 3x; its
    alpha is already tripled, so re-scaling it is a 3x error.
  - `swi_glu_mapping: Diffusers [value;gate] -> ComfyUI [gate;value]`. Already applied in
    this file. Applying it again swaps the swiglu halves, which is finite and wrong.
  - `training_scale: 0.0625 = training_alpha 8.0 / training_rank 128`. The per-tensor
    `.alpha` scalars are in the file, so read them rather than assuming alpha == rank.

## Foundation

Not H3-specific; these block everything else. **Done.**

1. **`LatentShape`** — `(channels, t, h, w)` plus an optional audio stream, in
   `pipeline.zig`. The still-image families are `t == 1, audio == null`, which is why
   they read the same as before. A flat latent is the visual stream followed by the
   audio one, `audioOffset` being the split, matching how ComfyUI packs it.
   - The `(lat_h, lat_w)` pair is NOT gone from the ~117 interior sites, and does not
     need to be: the per-family model code is right taking a plain `(h, w)` (SD's UNet
     has no temporal axis). The polymorphic surface is what generalized, so the 15
     model files were untouched.
   - `Session.spatialDownscale` / `latentShape` / `pixelFrames` replace a hardcoded
     `/ 8` at the call sites. H3's VAE is **16x**, so that hardcode was exactly the
     silent wrong answer this existed to prevent.
   - `latentPreviewInto` now takes the shape and a frame index. A video latent's
     per-channel stride is `t * h * w`; reading one with `h * w` previews channel 0 of
     several frames as several channels of one, which has plausible structure and
     wrong colours.
2. **`Clip`** beside `Image`: frames back to back in one allocation, interleaved
   stereo samples, an exact rational frame rate. Pure data, no libav.
3. **Family `minimax_h3`** in `Family` + `detectFamily`, probing `video_patch_proj` and
   `audio_patch_proj` **together** (ComfyUI's own test; one projection per stream is
   what makes it joint audio-video). `Component.decoder2` is the new kind for the audio
   VAE, the only family that answers it. Verified against the real 21 GB checkpoint,
   not just synthetic headers.
4. **A two-stream latent buffer.** **Done.** `generateClip` allocates one flat pack of
   `shape.elems()`, noises all of it, and Euler-steps it whole; `ClipDenoiser.predict`
   splits at `LatentShape.audioOffset` to convert the audio half in and out of the
   carried space, and `processLatentOut` unscales that slice at the end.

`Session.generate` still reports `error.FamilyNotImplemented` for this family, which is
correct rather than unfinished: a still image is not something H3 produces.
`generateClip` is the entry point.

## Wiring

`Session` now has the video-family stage surface beside the still-image one:

- `latentShape` / `pixelFrames` size a render; `encode` conditions it.
- `clipDenoiser` builds the per-render state (packed layout, trunk workspace,
  refined text) once, and `ClipDenoiser.predict` is one step over the packed
  two-stream latent.
- `generateClip` composes those with Euler, then `processLatentOut`, then
  `decodeClip`. No CFG branch: every H3 conditioning node emits ONE conditioning
  and takes no negative, so there is nothing to guide against.
- `decodeClip` runs both decoders and returns a `Clip`.
- CLI: `zig build run -- generate-clip --dit ... --vae ... --audio-vae ...`,
  writing PNG frames and a 16-bit WAV. Muxing is still owed.

**The audio carry lives in `ClipDenoiser.predict`.** The sampler's latent holds
the audio scaled onto the video schedule (`audio_scale = shift_video /
shift_audio = 4`); `predict` undoes it before the network sees it and converts the
velocity back afterwards, and `processLatentOut` unscales at the end. A sampler
never has to know about any of it.

## Stages

Each stage lands with tests. Ungated pure-op fixtures under the owning module's
`assets/`, gated real-checkpoint parity behind `-Dintegration` + `requireModelFile`.

1. **Foundation** as above. **Done.**
2. **Text encoder variant.** **Done**, for text-only prompts.
   `Config.qwen3vl_32b_h3` + `Variant.minimax_h3`: 50 of Qwen3-VL-32B's 64 layers,
   hidden 5120, 64 q over 8 kv heads, mlp 25600, theta 5e6, prefix `model.` (NOT
   krea2's `model.language_model.` -- this export puts the vision tower beside the
   language model at `visual.`).
   - **The final-norm rule was a biconditional that held by coincidence.** It asserted
     `appliesFinalNorm == (last tap == n_layers)`. Anima and H3 tap at the SAME place
     and disagree: Anima's `layer = "last"` applies the norm, H3 conditions on the raw
     state and its checkpoint has no `model.norm.weight` at all. Now an implication
     (applying the norm requires tapping at `n_layers`, not the converse), and the
     test enumerates the Variant enum so a new variant must answer.
   - **`loadWeight` only understood fp8's SCALAR `_scale`.** The sidecar has the same
     name for int8 convrot but means one scale per output ROW plus a rotation group.
     Reading the second as the first left `row_scale` null, which `ops.matmul` asserts
     on -- a panic several frames deep, not a wrong answer. `loadWeightNamed` now
     dispatches on dtype. `loadQK` refuses `permute_qk` on an integer weight rather
     than pairing row i's bytes with row j's scale.
   - Wired through `pipeline`: the H3 arm loads the trunk and the encoder, and
     `Session.encode` tokenizes the prompt RAW (no chat template, no special tokens,
     one pad token for an empty prompt, matching ComfyUI's `special_tokens={"pad":
     151643}`).

   **Owed:** numeric parity for this encoder. What is pinned is structural (config,
   prefix, tap, no-final-norm, per-row int8 scales) plus a shape/finiteness check on a
   real encode. A ComfyUI reference would mean running the 27 GB int8 model in torch,
   which is a real memory ask on this box; the krea2 variant IS numerically pinned and
   shares every code path except the config, so the risk is concentrated in the config
   constants, which the load test checks against the file.

   **Also owed:** vision blocks. fl2va/ref2va splice `<Picture i>:` labels and vision
   blocks into the presentation, which needs the Qwen3-VL ViT and, with it, **mrope**.
   Text-only is safe (Qwen's multimodal rope reduces to 1-D rope when all three
   position axes are equal), so t2va is complete and the image paths are not.
3. **`models/minimax_h3.zig`**, CPU reference. **Done.** Config detection, the packed
   layout builder, the weight loader and the forward.
   - The layout is pinned f64-exact against ComfyUI's own `PackedLayout` over 15 cases
     by `tools/gen_minimax_h3_fixtures.py`, plus two render-scale cases by checksum.
   - `Config.detect` is pinned against the real 21 GB checkpoint (gated).
   - The **forward** is pinned against ComfyUI's own `MiniMaxH3Model` built at a toy
     width by `tools/gen_minimax_h3_forward.py`, to a relative L2 of ~3e-7. The toy
     width keeps `inner != hidden` (64 vs 32, as the real model's 7168 vs 5376) so a
     square `out_proj` cannot pass by accident.
   - `Workspace.bytesFor` reports what a shape will cost before allocating. The CPU
     path is a reference implementation: at the default render its activations are
     over 4 GB, so a caller must be able to refuse.

   Three generator notes, each of which cost a cycle:
   - ComfyUI resolves `optimized_attention` at IMPORT time and checks xformers before
     pytorch. xformers has no CPU kernel at all, and the CLI flags do not help because
     the model module binds the name by value. Rebind
     `comfy.ldm.minimax.model.optimized_attention`.
   - The fused RMSNorm+rope kernels are in-place and refuse any tensor with
     `requires_grad` set. `no_grad` and `inference_mode` do not clear that flag on a
     parameter; `requires_grad_(False)` does.
   - **The first version of this fixture had no teeth.** With `randn * 0.05` weights
     throughout, swapping the swiglu halves moved the output by 3e-7 against a 1e-5
     threshold, i.e. not at all. Two causes: RMSNorm scales drawn around ZERO
     annihilate the signal, and activations that never leave silu's near-linear region
     make `silu(a)*b` and `silu(b)*a` agree to second order. Fixed by initializing the
     way real weights are distributed (norm scales centred on 1, projections
     fan-in-scaled), which moved the swiglu sensitivity to 37%. The generator now
     asserts that sensitivity itself.

   Breaks confirmed to fail against the fixture: swiglu halves swapped, adaLN
   tag/timestep transposed (15.7%), output negation dropped (200%), audio interleaved
   instead of channel-major (4.2%), and rope over the whole head instead of the first
   24 dims.
4. **ViT3D video decoder.** **Done** on CPU (`models/minimax_h3_vae.zig`) and CUDA
   (`minimax_h3_vae_cuda.zig`, 420 s -> **4.9 s**, matching the CPU path to 1.3e-5).
   The device port needed two things the DiT's did not: `opDeinterleave3`, because
   `to_qkv` is fused PER HEAD so the planes are not row ranges; and the SD family's
   head padding, because `opAttnTC`'s P@V GEMM tiles at 128 and this VAE is 32 heads
   of **64**, which launches a zero-sized grid rather than computing a wrong answer.
   `decodeTemporalWith` takes the per-window decode as a callback, so the chunking and
   blending have one implementation whichever backend runs the window.
   CPU path pinned to ~1.2e-7
   against ComfyUI's own `ViT3DDecoder` by `tools/gen_minimax_h3_vae.py`.
   Several conventions DIFFER from the DiT in the same checkpoint family, which is the
   trap: `to_qkv` is fused **per head** (`[h0 q|h0 k|h0 v|h1 q|...]`, where the DiT is
   `[all q|all k|all v]`); the Q/K norms are **weightless**; position ids are per-axis
   normalized to [-1, 1] rather than area-normalized; residuals carry a per-channel
   **LayerScale**; and the head is a LayerNorm where every other norm is RMS.
   `head_dim` (64) and the patch geometry (16 spatial, 4 temporal) are constants in
   `Config.detect`, because `dim` and `proj_out` state only the products; the fixture
   uses the REAL values so it validates the split the real model uses.
   **Temporal chunking is implemented** (`Temporal`), and it turned out to be
   load-bearing rather than an optimization: a whole-volume decode gives
   `t * patch_t` frames where the reference gives fewer (2 latent frames -> **5**
   pixel frames, not 8; 37 -> **124**, not 148), and it blends overlapping windows,
   so the content differs too. Those counts are exactly what
   `minimax_h3.framesForLatentT` predicts from the DiT's own time axis, and a test
   asserts the two halves agree for every valid length -- they compute it by
   completely different arithmetic, so that agreement is a real check.
   **The SPATIAL tiling is load-bearing too, and this cost a real bug.** The
   reference's VAE has `tiling = True` by default, so `_adaptive_decode` ALWAYS goes
   through `tiled_decode`: 256 px tiles with a >=64 px overlap, blended. The ViT3D
   therefore never sees more than `tile / patch` = 16 spatial tokens per axis.
   Decoding a whole frame in one pass ran the trunk at an extent it is never given,
   and the result was a visible 16 px grid wherever the image had detail and smooth
   where it did not. A 256 px frame is exactly ONE tile, which is why 256x256
   renders looked right and 512x512 did not.

   Ruled out first, each by measurement rather than argument: the device VAE
   matches the host at exactly the render's 7173-token sequence (5e-6); f32
   attention gives a bit-comparable result, so it is not the tensor-core path's f16
   scores; the residual stream peaks at **max|x| 8.2**, nowhere near f16's 65504;
   the DiT device path does not degrade with sequence length (0.0029 at 150 rows,
   0.0027 at 534); and 30 steps looks the same as 6, so it is not undersampling.

   `Spatial.splitTiles` mirrors the reference's placement (320 px -> 2 tiles at
   0/64 overlapping 192; 512 px -> 3 tiles at 0/128/256 overlapping 128) and a
   fixture pins the tiled decode's pixels. **Every earlier fixture case was below
   the tile size, i.e. a single tile, which is exactly why they passed against a
   non-tiled decode** -- the generator had `tiling = False` and that was the hole.

   One performance trap it introduced: the device session is per SHAPE, and the
   tiles are a different shape from the window. Sizing it for the window made every
   tile miss and fall back to the host, a silent 100x. It is sized per tile.
5. **Audio VAE decoder (BigVGAN).** **Done** (`models/minimax_h3_audio.zig`), pinned to
   ~1.1e-6 by `tools/gen_minimax_h3_audio.py`. The new 1-D op family (Conv1d with
   dilation/groups, ConvTranspose1d, replicate padding, SnakeBeta) lives in that file
   rather than `ops/`; hoist it if a second consumer appears.
   - The kaiser-sinc filters are **stored in the checkpoint**, so they are loaded, not
     recomputed. That removes the whole kaiser-window/sinc derivation.
   - **alpha/beta are in LOG scale** (`x + sin(exp(a)x)^2/exp(b)`).
   - Dilation and the upsample rates are **not stored**: `convs1` cycles (1, 3, 5) and
     the rates are recovered from the kernel sizes (`k = 2u`, except `u = 5` takes
     `k = 9`). The padding is coupled to both, and a mismatch stops the block being
     length-preserving.
   - Each stereo channel decodes independently through the same mono vocoder; the
     fixture asserts the two channels actually differ, since decoding one and copying
     it would pass every shape check.
6. **Sampler.** **Done.** Flow with the dual shift and the audio carry;
   `.discrete_flow = shift` is the base schedule and `minimax_h3.Shifts` carries the
   pair. Euler only so far, which is what the reference's H3 workflows use; the SDE
   samplers would need the Brownian tree to reach the audio slice too.
7. **MP4 output.** **Done.** `lib/video/av_helper.c` + `src/av.zig`, linked into the
   diffusion EXECUTABLE only, so the library tier stays pure Zig. H.264 + AAC, and the
   AUTOMATIC1111 `parameters` block in the container metadata. `--frames` still writes
   a PNG sequence plus a WAV when you want to inspect individual frames.
   - **`buildA1111Params` moved from `gui/diffuser.zig` into `pipeline`**, with the GUI
     re-exporting it. Two copies of a format whose entire purpose is round-tripping is
     the drift that makes a reader re-render the wrong image; the MP4 tag and the PNG
     text chunk now come from one builder.
   - ⚠️ **MP4 silently drops unknown metadata keys.** The first muxed file had no
     `parameters` tag at all and reported no error: isom keeps only the handful of keys
     it defines. `movflags use_metadata_tags` puts arbitrary keys in a udta atom.
   - Verified by probe: h264 512x512 24 fps 22 frames (0.875 s) alongside aac 32 kHz
     stereo (0.928 s), the tag round-tripping with its newline, and a frame extracted
     back out of the container.
8. **LoRA sidecar**: **done**, `models/lora.zig` (loader + host apply) and
   `models/lora_cuda.zig` (device apply). Architecture-independent: it keys on
   `Weight.tag`, so any family whose loader sets one can attach a LoRA.
   `--lora PATH [--lora-strength F]` on `generate-clip`.
   - `y = W x + s B (A x)`, never merged. The trunk is int8 convrot, so a merge means
     dequantizing 20 GB, adding the delta and requantizing, and the result is only
     reproducible if the original quantizer bake is. A sidecar has no round trip and
     leaves `strength` a runtime dial.
   - `minimax_h3.Lin` pairs each weight with its sidecar so the weight is reachable
     only as `.w`. That is the structural answer to the hazard CLAUDE.md names: a
     sidecar applied at some GEMM call sites and not others is a plausible wrong render
     with no error anywhere. `matLin` is the single host path.
   - The fused qkv's `B` is **block diagonal**, detected on the data (exact zero off
     the diagonal, early-out on the first nonzero) rather than by name. Splitting it
     cuts the B GEMM and the resident bytes by 3x; the A GEMM is the same either way,
     so the fused target comes out ~2.1x cheaper. Measured on the real turbo LoRA:
     **1319 MB resident instead of 1866**.
   - The device asks for an output **row range**, not a whole target, because a trunk
     holds a fused linear's output as separate buffers (qkv in three planes, `fc1` as
     gate and value). Rows of `B` are the output's columns, so a row slice of `B`
     produces exactly the piece wanted with no de-interleave.
   - Validated three ways: `lora.zig`'s fixture test reproduces ComfyUI's own MERGED
     LoRA to under 1e-6 at two strengths (`tools/gen_lora_fixtures.py` executes
     `comfy.lora.load_lora` + `LoRAAdapter.calculate_weight`, and asserts that
     dropping the `/rank`, tripling the fused alpha and transposing `A` each move the
     corpus); `lora-cuda-test` puts the device apply against the host one at the real
     factor shapes; and a render.
   - ⚠️ **`s = alpha / A.rows`, from the file's own shapes.** The fused qkv factor has
     A with 384 rows and an alpha already multiplied by 3, so `24/384` and the
     per-block `8/128` are the same 0.0625 only if the divisor is the stored row
     count. Using the per-block rank on the fused alpha is exactly a 3x error.
   - Device residual is **2.3e-3** relative, and the control row accounts for all of
     it: 1.66e-3 from rounding the activation to bf16 for the A GEMM and another
     1.66e-3 for the B GEMM, which add in quadrature to 2.35e-3. That is below the
     int8 base path's own ~4e-3, so the sidecar is more accurate than the GEMM it
     corrects.
   - ⚠️ **The accumulate cost more than the GEMMs it served.** Materializing the
     delta into a scratch plane and adding it is three extra passes over an f32
     `[m][n]` plane on top of the GEMM's own output write, which at H3's widths is
     ~70 GB per step. `opGemmBf16Acc` folds it into cuBLASLt's epilogue with
     `beta = 1` and `C == D` (free, since `ltRun` already passes one buffer for
     both), which also drops the scratch plane entirely (205 MB at the native
     canvas). The hand-PTX `hgemm` writes its C tiles unconditionally, so
     `zig-cuda` keeps the scratch-plus-add route; `Workspace.fused` picks once per
     session and `TP_LORA_NO_FUSE` A/Bs them.

     Measured at 512x512 / 22 frames, `--backend cuda`, interleaved within a round
     (see the warning below about why only within-round pairs count):

     | | s/step |
     |---|---|
     | no LoRA | 0.80 |
     | LoRA, both GEMMs, accumulate skipped (`TP_LORA_SKIP_ADD`, wrong render) | 0.86 |
     | LoRA, fused accumulate | 0.93 |
     | LoRA, unfused | 1.01 |

     So the GEMMs are +7%, the fused accumulate takes the total to +16%, and
     leaving it unfused would make it +26%. The FLOP-count estimate that motivated
     this feature said 3-4%; it was counting only the GEMMs, which is the mistake
     the accumulate row exists to prevent repeating.
   - The LoRA's own step times drift by ±30% between rounds while the no-LoRA
     baseline holds 0.78-0.81. The asymmetry is the likely signature of VRAM
     pressure: 1.3 GB of resident factors on top of a ~20 GB streamed trunk on a
     24 GB card. That makes the block-diagonal split worth more than its FLOP
     saving suggests, and it means a LoRA figure from this box carries a wide band.
9. **GPU ports.** All three CUDA ports are **done** — trunk, video VAE, audio VAE.
   `minimax_h3_gpu.zig` (Vulkan) is not started.

   **The audio VAE** (`minimax_h3_audio_cuda.zig`): **47.8x** at 37 latent frames
   (3591 ms -> 75 ms), 60x at 62, which takes a 22-frame clip's audio decode from ~9 s
   to ~0.2 s. Validated by `minimax-h3-audio-cuda-test` against the CPU decode on real
   weights, reporting a per-sample maximum and an error histogram, not just a norm.
   - Four new kernels: `im2col1d`, `aa_up_snake`, `aa_down`, `convt1d_ca`.
   - **Signals are CHANNEL-LAST on the device**, unlike the reference's planar
     `[ch][len]`. That is what makes every kernel coalesced (a warp covers consecutive
     channels at one time step) and it removes the transpose the CPU im2col pays on
     both sides of its GEMM. The conv weights are permuted once per session to the
     matching `(tap, in_ch)` column order; pairing the two the other way convolves
     every tap against the wrong channel.
   - **The anti-aliased activation collapses from seven ops to two kernels.** Read as
     a gather rather than the reference's pad -> transposed conv -> slice -> scale ->
     snake -> pad -> conv, the replicate padding is an index clamp and the slice is an
     index offset, so the whole first half is one pass with one `2 * len` intermediate.
   - **Its transposed convs are gathers too**: output `t` reads position `t + pad`,
     whose contributing taps are exactly `j == (t + pad) mod stride`, i.e. `k / stride`
     taps (2 at every stage here). A scatter would need atomics.
   - ⚠️ **The GEMMs run in f32, and the f16 tensor-core route was worse on BOTH axes.**
     Against the f32 CPU reference: f16 is 2.4e-3 relative with a 8.3e-3 worst sample
     (about -42 dB, 74% of samples past 1e-4), which for a vocoder is an audible noise
     floor; f32 through cuBLASLt is 9.5e-6 / 2.8e-5, under 16-bit PCM's own quantum.
     And at the real shape f16 measured 82 ms against f32's 75 ms, because the
     per-conv weight-pad and activation-convert passes cost more than the tensor cores
     win at these widths. `opMatmulF32Lt` uses `COMPUTE_32F`, deliberately NOT
     `_FAST_TF32`: TF32 keeps 10 mantissa bits, the same precision as the f16 path it
     exists to avoid. `zig-cuda` has no tiled f32 GEMM and keeps the
     one-thread-per-output kernel — correct, 6.9x instead of 47.8x.
   - The f32-vs-f16 split is also what PROVED the four kernels correct: with f32 GEMMs
     the whole decode lands at 9.5e-6, so the entire original gap was precision and
     none of it was a defect. `TP_H3_AUDIO_F16=1` reproduces it.

   **The trunk** (`minimax_h3_cuda.zig`): **129x** over the CPU trunk at 150 packed
   rows (230 s -> 1.8 s). Validated by `minimax-h3-cuda-test` against the CPU forward
   on real weights.
   - The host keeps the patch projections, the adaLN projection, the token refiner and
     the output heads: several have shapes the device GEMMs refuse (`adaln_proj` is 8
     columns wide; the heads are 96 and 32 rows against `opI8Gemm`'s 128-row floor), and
     together they are under 0.01% of a step.
   - Per-segment modulation is a `rmsMod`/`gatedAdd` launch per segment on an offset
     view, because segments are contiguous. No new kernel.
   - `rms_mod_par` applies NO norm weight and NO `1 +`; the host folds
     `norm.weight * (1 + scale)` into its premul slot.
   - The fused `qkv_proj` and `fc1` are split by ROWS into `Weight` views over the same
     mapping (a row range of a row-major weight is contiguous and its per-row scales
     slice with it), so there is no de-interleave kernel and no copy.
   - Device buffers are padded to 128 rows: `opI8Gemm` writes the activation count
     ROUNDED UP, so an exactly-sized buffer is written past its end.

   ⚠️ **Finding this port cost a real backend bug.** `buildPrep`'s FWHT butterfly count
   was `ngroups * 64 / 256`, which truncates whenever `cols % 1024 != 0`, leaving the
   tail of every activation row in its unrotated basis: no error, no assert, a GEMM in
   the wrong basis. Every width in the engine was a multiple of 1024 until H3's 5376
   hidden (21 groups, needing 5.25 iterations, getting 5) at **22%** error. See ZIG.md
   and BACKEND.md 2E. `krea2`/`anima`/`bqdec` device tests are byte-identical before and
   after, since their widths never truncated.

   Accuracy against the CPU forward, at 150 rows: video 0.009, audio 0.061 relative L2
   at depth 50, flat at 0.003 through 45 blocks. Isolations that rule out a defect: one
   qkv GEMM alone is 0.0088 (the expected W8A8 level), and the naive attention moves the
   total by under 6%. The residue is the last block, whose `fc2` scales average 3x
   smaller than any other's. **Owed:** a render compared against ComfyUI, which itself
   runs this checkpoint W8A8 and so may agree with the device path more closely than the
   f32 host reference does.
10. **Reference path**: Qwen3-VL ViT with deepstack, the 3D causal CNN encoder, the DAC
    audio encoder, refs/keyframes/denoise masks in the layout. **Done** except the
    device vision encoders and a CUDA port of the two VAE encode sides; denoise
    masks are still owed.

    The layout half has been done since the foundation: `PackedLayout.build` already
    takes `keyframes` and `refs`, places their rows and positions, and is pinned
    f64-exact against the reference. What was missing is everything that turns a
    picture into those rows.

    - **The video VAE's ENCODE side is done** (`models/minimax_h3_vae_encode.zig`),
      matching ComfyUI to under 1e-5 on a single frame, a frame wider than the
      256 px tile, and a 5-frame clip. A sibling file rather than part of the decode
      one: a six-level causal CNN and a ViT3D share nothing but a checkpoint, the
      `Spatial` tiling and the latent statistics.
      - ⚠️ **The single-frame path is not a different calculation.** The reference's
        `causal_zero` branch is an optimization -- the causal front pad is all zeros,
        so it truncates the temporal taps rather than convolving frames it knows to
        be zero. Written as a gather it is one formula, and only the LAST temporal
        tap lands on a single frame. Giving it its own length formula yields ZERO
        output frames.
      - ⚠️ **GroupNorm statistics are per FRAME**, and a single-frame encode is blind
        to that axis. The fixture needs a clip case or the convention goes untested.
      - A 3-D convolution is a GEMM and the weight layout is already its B matrix.
        The naive loop ran at ~0.2 GFLOP/s, which put one full-resolution reference
        image at tens of minutes; banded im2col took the unit test 22 s -> 3 s.
        Banded because the patch matrix duplicates each input 27x, which un-banded
        is 4.7 GB for a single level-0 convolution at 1344x768.
      - **The CUDA port is done** (`models/minimax_h3_vae_encode_cuda.zig`):
        47.9 s -> 1.11 s at 17 frames of 256x256, and the 4.35 GB host peak becomes
        1.8 GB of VRAM. Validated by `minimax-h3-vae-encode-cuda-test` at rel L2
        ~2e-6, on a CLIP rather than a single frame because GroupNorm statistics are
        per frame and one frame is blind to that axis.
        - **Only `encodeMoments` moved.** The clip chunking and the spatial tiling
          above it are intricate, already pinned, and pure bookkeeping, so the CPU
          module takes a `Moments` function pointer and both backends share one copy
          of them. The hook's workspace GROWS per call, because tiling and chunking
          each hand it something smaller than the render.
        - Volumes are CHANNEL-LAST there, which buys more than coalescing:
          `opGroupNorm` is already channel-last and already fuses the SiLU that
          follows every norm here, so the norms need no kernel of their own.
        - ⚠️ **A BANDED conv cannot have its destination alias its source.** Band
          0's GEMM overwrites input that band 1 has not read yet, which is correct
          for anything small enough to fit one band and wrong for everything larger.
          Both ports now assert it.
        - ⚠️ **An asymmetric reflect pad must not be folded into the input EXTENT.**
          Telling the gather the volume is one row taller so it can reflect into the
          pad reads a row the buffer does not have: an illegal access, not a wrong
          number. The high pad extends the OUTPUT; the reflect is always against the
          real extent, which is what the reference does too.

    - **The ViT is pinned but not ported** (`tools/gen_minimax_h3_vit.py`).
      ⚠️ **`models/vit35.zig` cannot be reused, despite being the same tower on
      paper** (27 blocks, 1152 wide, fused qkv, 4304 FFN, 48x48 position table, and
      even the same `merger.norm` under llama.cpp's `v.post_ln` name). It was ported
      from llama.cpp's clip.cpp and TWO conventions diverge from the HF lineage H3's
      checkpoint actually is:
      - **The position-embedding interpolation.** llama.cpp resamples the 48x48
        table with `BILINEAR | ANTIALIAS`, `align_corners=false`; the reference does
        `linspace(0, 47, n)` with plain floor/ceil bilinear weights, i.e.
        align_corners TRUE and no antialias. Different numbers at every grid that is
        not natively 48x48, which is every real image.
      - **The patch embedding's temporal pair.** `temporal_patch_size` is 2 and
        llama.cpp SUMS the two 16x16 convolutions, which is right only when both
        temporal frames are the same still. H3's reference VIDEOS feed two DISTINCT
        frames per block (`process_video_block`), so the sum is wrong there.

      So the H3 tower needs its own forward. Checking first is what turned a
      "reuse it" assumption into two named defects; a correct port of the wrong
      convention passes every test.

      **`models/minimax_h3_vit.zig` is done**, matching the reference to under 1e-5
      on the tower output and all three deepstack features. The rope DOES agree
      between the lineages, so `vit35.applyVisionRope` is reused.
      - ⚠️ **The patches arrive ALREADY in merged block order.**
        `process_qwen2vl_images` permutes on the way out, and
        `fast_pos_embed_interpolate` returns positions in that order too. So the
        forward permutes NEITHER the tokens; what it permutes is the position table,
        interpolated row-major and then reordered to match. Permuting the tokens as
        well applies the 2x2 merge twice and pairs every token with another patch's
        position: 0.75 relative error, no crash, and it looks like a bad port of
        something else entirely.
      - ⚠️ **The block MLP is gelu-TANH and both mergers are erf gelu.** Same name,
        two functions, and the gap is small enough to read as noise.

    - **The presentation is half done.** The LAYOUT side has landed: `Kind.text_vision`
      (text-encoder rows that carry the VIDEO tag), and `PackedLayout.build` takes
      `vision_spans` and splits the text region into alternating `.text` /
      `.text_vision` runs. Malformed spans are refused (`error.BadVisionSpan`)
      because the runs must TILE the region: the modulation loop walks segments, so
      a gap leaves rows unmodulated and an overlap modulates them twice. A vision
      run takes the SAME timestep as the prose it sits in and differs only in tag.

      The ENCODER side has landed too:
      - `ops.rope.mropeInterleavedFreqs` -- Qwen3-VL's interleaved multimodal rope,
        three axes sharing one `head_dim / 2` ladder ROUND-ROBIN rather than in
        contiguous sections.
      - `minimax_h3_vit.mropePositions` -- the `[3][seq]` position construction.
      - `qwen3.TextEncoder.encodeVision` -- an entry point taking a `Vision`
        payload (spliced embedding rows, mrope positions, deepstack features and
        their injection spans). `encode` is now a wrapper passing an empty one, so
        the five other consumers of this encoder are untouched by construction.
      - Both mrope halves match the reference exactly (`tools/gen_minimax_h3_mrope.py`).

      And the PRESENTATION plus its threading:
      - `models/minimax_h3_present.zig` builds the labelled prompt -- reference
        labels, `<|vision_start|>` + placeholder rows + `<|vision_end|>` blocks,
        then the prompt -- and is the ONE place the token ids, the tag spans, the
        deepstack injection spans and the mrope `ImageSpan`s are derived from.
        Deriving them separately is how they drift.
      - `EncodeOptions.h3_refs` carries the reference items and `Cond.h3_tag_spans`
        carries the spans back out to `PackedLayout.build`, because only `encode`
        knows where the blocks landed.
      - `Session.runQwen3Vision` keeps an empty payload on the normal CUDA/Vulkan
        dispatch and forces a non-empty one to the CPU, loudly: the device encoders
        have neither deepstack nor mrope, and running them would produce a
        conditioning with no vision in it rather than a slow correct one. **Owed:**
        the VULKAN one. **The CUDA one is done** (`qwen3_cuda.encodeVision`): the
        block rows are pasted before the upload, mrope replaces the 1-D rope table,
        and DeepStack is added per span with an offset `opAdd` -- an injection span
        is a contiguous row run, so it needs no kernel at all. Validated by `te-test
        TP_TE_VISION=1` at 1.44e-4 device-vs-CPU on a real block-quant encoder
        (text-only control 1.29e-4), and the teeth confirmed by disabling the device
        injection, which fails at 4.19e-2.
      - ⚠️ **CORRECTION.** An earlier note here bounded the vision port at "+7.8 s
        of a 66 s render". That was wrong: `supportsWeights` refused H3's encoder on
        EVERY backend, because `qwen3_cuda`'s encode GEMM was fp8/bf16/block-quant
        only and this checkpoint is int8-convrot. Both runs in that comparison were
        on the CPU, so the 7.8 s was the ViT plus the reference's VAE encode, not a
        device-vs-CPU encode difference. The H3 text encoder has never run on the
        device, and doing so is worth the whole ~29 s encode.
      - The int8 arm is in `wgemm` (`opI8Prep` then `opI8Gemm`, one prep shared by
        q/k/v as the DiT does), and the prep's width ceiling is gone: it stages the
        row in GLOBAL memory when it will not fit shared. **60x where the weights
        fit** -- 25 layers of H3's encoder go 7421 ms -> 124 ms at rel L2 1.72e-2.
        - ⚠️ **VRAM is what keeps this encoder on the CPU, not kernels.** 23250 MB
          of int8 weights against 21451 MB free on a 3090, and past the fit the
          per-layer re-upload makes the device SLOWER than the host: 50 layers take
          20840 ms against an 18384 ms CPU encode, while 25 take 124 ms. The cliff
          is that sharp. `supportsWeightsOn` gates on measured free VRAM, so the
          earlier "~29 s prize" is real but not collectable on a 24 GB card.
        - The isolation that separated the two questions is `te-test TP_TE_TAP=n`, a
          DEPTH sweep. One layer already diverges 6.5e-3 and an isolated int8 GEMM
          is 0.0088, so the 1.8e-2 at 50 layers is int8's floor, not a defect -- and
          the same sweep is what exposed the residency cliff between 25 and 50.
        - The prep's global staging is bit-identical to the shared one where both
          work: `minimax-h3-cuda-test TP_H3_GLOBAL_PREP=1` forces it at a fitting
          width and measures rel L2 exactly 0. That matters because it is shared
          machinery -- every int8/int4 GEMM in the engine goes through this prep --
          and a device-vs-CPU tolerance could not see an address-rewrite bug under
          int8's own ~1e-2 floor.
      - **NUMERIC parity is pinned** (`tools/gen_minimax_h3_te.py`), against
        ComfyUI's own `Llama2_` under `Qwen3VL_32BConfig` at a toy width, to under
        1e-5 on TWO cases: text-only (every t2va render) and a spliced vision block
        with mrope and deepstack. This is the first numeric fixture `qwen3.zig` has
        had at all, so krea2, Z-Image and Anima gain from it too -- they share the
        trunk and differ only in config, tap list and final norm.
        - ⚠️ **`head_dim` is 128 and is NOT `hidden / n_heads`**: the real model is
          5120 wide over 64 heads, so q/k/v project to 8192. The toy keeps that
          asymmetry (128 wide, 256 inner), because a square encoder passes an
          o_proj transpose by accident.
        - Teeth confirmed by breaking the implementation: theta 1e6 instead of 5e6
          fails at 0.109, dropping deepstack at 0.226, dropping mrope at 0.230. The
          fixture also measures that applying a final norm moves the answer 0.342,
          which is the convention this variant alone inverts.
        - Fewer deepstack features than layers (2 over 3), so the rule that
          injection STOPS after the last feature is pinned rather than coinciding
          with the layer count.
        - ⚠️ **`Llama2_.forward` mutates the `embeds` you hand it and returns that
          tensor** -- its residual adds are in place. Every reference call needs its
          own clone, and a stored output needs `.clone()` not `.contiguous()`: on an
          already-contiguous view the latter is a VIEW, so a later sensitivity check
          overwrote the tensor it was being compared against and reported a
          difference of exactly zero. That zero is what caught it.
      - ⚠️ **Reference ordinals count per TYPE, not globally.** Image, audio, image
        is `<Picture 1>`, `<Audio 1>`, `<Picture 2>`. A global counter gives
        `<Picture 1>`, `<Audio 2>`, `<Picture 3>` -- fluent, and referring to things
        the prompt never labelled.
      - ⚠️ **An audio reference gets a label and NO block**; it never enters the
        vision tower, its rows enter the DiT through the layout.
      - ⚠️ **A reference video's soundtrack label comes BEFORE its video label**,
        matching the order the DiT packs their rows in.
      - The placeholder rows under a block are real pad ids, not a sentinel: the
        encoder embeds every id before pasting the vision rows over them, so an
        out-of-vocabulary marker would fail the lookup instead of being replaced.
      - Verified unchanged for t2va by re-rendering a stored baseline (256x256,
        22 frames, 8 steps, seed 0) and byte-comparing every frame and the WAV.
        That check matters more than usual here: `encode` is now a wrapper on
        `encodeVision` for FIVE other families, and the krea2 conditioning parity
        test self-skips on this box, so nothing that runs covers it.

      Conventions, all pinned:

      - ⚠️ **One block, TWO spans, differing by a row at each end.** The modality
        TAG span is widened by one on each side so the flanking
        `<|vision_start|>` / `<|vision_end|>` markers are tagged with the image;
        the DEEPSTACK injection span is NOT widened and covers only the embedding
        rows. `minimax_h3.VisionBlock` derives both so a call site cannot reach for
        the wrong one.
      - **Vision blocks switch the LLM to mrope**, and only then:
        `build_image_inputs` returns no position ids when there are no images, so
        the text-only path is correctly plain 1-D rope, and `positions == null` is
        what selects it.
      - ⚠️ **The three rope axes are INTERLEAVED, not sectioned.** `rope_dims`
        `[24, 20, 20]` over 64 slots reads naturally as three contiguous blocks and
        is not: slot `k < 3 * dims[1]` takes axis `k % 3` and the rest take axis 0,
        so axis 0 holds slots 0,3,..,57 PLUS the tail 60..63. The two readings
        differ by 0.13 with a real image and agree EXACTLY on a text-only prompt,
        which is when nothing would notice.
      - The mrope construction (`qwen2vl_mrope_position_ids`), which is intricate
        enough to port rather than re-derive. Per image at `[start, end)` with
        merged grid `gh/2 x gw/2`:
        - axis 0 (T) is CONSTANT across the block at `start + offset`;
        - axis 1 (H) is the merged ROW index, each value repeated across a row;
        - axis 2 (W) is the merged COLUMN index, the range tiled;
        - the block consumes `max(gh, gw)/2` positions on the timeline whatever its
          token count is, text after it resumes at `start + len_max + offset`, and
          `offset` accumulates `len_max - (end - start)` per image. So the timeline
          and the token index DRIFT apart across several references.
        - axis 1 outer / axis 2 inner means mrope reads the merged tokens in
          ROW-MAJOR order, which is what merging the ViT's block order gives.

    - **DeepStack**, from the reference: features are taken from the OUTPUT of ViT
      blocks `[8, 16, 24]` and injected into LLM decoder layers `0, 1, 2`, added at
      image-token positions only. Note the asymmetry -- collected deep, injected
      shallow. The features from several images are CONCATENATED in index order and
      the position mask selects them in sequence order, so the two orders have to
      agree.
      ⚠️ **The two mergers normalize at DIFFERENT widths, and it is one set of
      parentheses.** The main merger is `norm(x).view(-1, merge_dim)`, LayerNorm over
      the PRE-merge 1152; the deepstack merger is `norm(x.view(-1, merge_dim))`,
      LayerNorm over the POST-merge 4608. The checkpoint says so out loud
      (`merger.norm` is [1152], `deepstack_merger_list.N.norm` is [4608]), so reading
      them the other way is a shape error only by luck.

    - **The audio VAE's ENCODE side is done** on CPU and CUDA
      (`models/minimax_h3_audio_encode{,_cuda}.zig`), the device port at rel L2
      ~2e-6 and 16-22x at a realistic reference length. Both ports are about EVEN on
      the hand-PTX arm, which has no tiled f32 GEMM, and both stay in f32 there
      rather than trading the 42 dB the decode side measured f16 to cost.
      The CPU side (`models/minimax_h3_audio_encode.zig`),
      matching ComfyUI to under 1e-5 at a toy width. A sibling of the decode file
      for the same reason the video pair are siblings: a DAC down-sampler and a
      BigVGAN vocoder share `Conv1d`, the im2col GEMM and a checkpoint, and nothing
      else. Every activation and every block shape differs.
      - ⚠️ **`Snake1d`, not `SnakeBeta`.** The encoder stores alpha LINEAR and uses
        the same parameter as beta; the decoder stores alpha and beta separately and
        in LOG scale. One activation name, three differences, and substituting one
        for the other is finite and wrong (0.58 relative here).
      - ⚠️ **The residual dilations are 1, 3, 9** in the encoder and 1, 3, 5 in the
        decoder.
      - ⚠️ **The posterior head's attention is CAUSAL**, and its output is the MEAN
        OVER HEADS pooled along the FEATURE axis (256 -> 32) down to the latent
        width. Pooling TIME instead is the obvious misreading and gives a plausible
        shape. Non-causal attention moves the answer by 0.14, so it is not free.
      - ⚠️ **The head count is not in the checkpoint.** The fused qkv is one
        `[3 * 2048][2048]` matrix whatever the split is, and the split changes the
        scale, the softmax grouping and the pooling's input width. It is an
        `AudioEncoder.Options` field defaulting to 8, not something derived.
      - ⚠️ **The reference NODE transposes the waveform before the VAE**
        (`waveform[:1].movedim(1, -1)` on a `[B, C, L]` tensor), which feeds
        `encode` a `[1, L, C]` and makes it read the sample count as the stereo
        width. `MiniMaxH3AudioVAE.encode`'s own docstring says `[B, 2, L]`, and that
        is what is implemented: the node's transpose cannot be what the model was
        trained on. A reference is a piece of code and can be the thing that is
        wrong.
      - `latents_mean/std` and the `logs_proj` / `mean_proj` pair are shared with the
        decode side; only `mean_proj` is read, and no sampling happens.
      - Verified on the REAL weights by a gated round trip: two different tones, one
        per stereo channel, encoded and then decoded back through the independently
        written BigVGAN. Each channel's own tone dominates the other's by more than
        10x. A wrong dilation cycle, a pooled time axis or a swapped norm all
        produce a latent that decodes to *something*; only a surviving pitch says
        the two halves agree about what a latent means.

    - **Waveform I/O and resampling** live in `core/audio.zig`: a WAV reader (PCM
      8/16/24/32, IEEE float 32/64, `WAVE_FORMAT_EXTENSIBLE`, unknown chunks
      skipped), the 16-bit writer moved out of the CLI, and torchaudio's
      `functional.resample` at its defaults, which is the filter every reference
      implementation feeds its audio VAE through.
      - ⚠️ Both rates are divided by their GCD FIRST and every constant follows from
        the reduced pair; `rolloff` sits inside the window argument, the sinc
        argument AND the output scale; the polyphase output is TRANSPOSED before
        flattening (flattening the other way interleaves the branches and shifts the
        pitch, 1.34 relative).
      - The kernel is computed in f64 and narrowed once. **Read the control row:**
        torchaudio computes it in f32, and for 44.1k -> 32k (441 -> 320 phases, whose
        near-zero taps have little significance left in f32) its own f64 and f32
        kernels differ by 9.86e-6 relative -- exactly the gap our kernel shows
        against the f32 one. The fixture carries both so the test compares against
        the f64 filter and records the f32 gap as the reference's floor.

    - **The reference DATA path** is `Session.h3EncodeRef` / `h3EncodeKeyframe` /
      `h3EncodeRefVideo` / `h3EncodeRefAudio`, all producing both halves of a
      reference at once so the LLM's features and the DiT's condition rows cannot
      describe differently resized inputs. `Cond.h3` carries the layout shapes,
      `cond_video`, `cond_audio` and the tag spans out of `encode`, which is the one
      place that processed the references. CLI: `--ref-image`, `--ref-size`,
      `--ref-video`, `--ref-video-audio` (paired positionally with the preceding
      `--ref-video`), `--ref-audio`, `--first-frame`, `--last-frame`.
      - ⚠️ **A reference video's audio and video rows are ONE block.** The audio
        rows pack immediately before the video's and share its cursor origin, and
        they take their width coordinates from the VIDEO's own latent grid -- not the
        target's, which is where a STANDALONE audio reference gets them. Same
        segment kind, same row count, different positions.
      - ⚠️ **The reference encodes the whole soundtrack even after cropping the
        frames** to a legal clip length, so a reference video's audio can be longer
        than its video. The layout handles it (a block's time span is the max of its
        two streams); we match the reference and warn when they diverge by more than
        four latent frames.
      - Keyframe AUDIO (`Keyframe.audio_t`) is placed by the layout but no caller
        can request it yet.
      - Both VAE encoders bound their host peak by banding the im2col and giving
        each level or stage its own arena; `peakBytesFor` reports it before the work
        starts. The audio encoder's first residual stage runs 64 channels over the
        FULL sample count, which un-banded is 573 MB of column matrix for ten
        seconds and grows linearly with the reference's length.
      - **A reference soundtrack steers the generated audio, verified by render.**
        Same prompt, seed and settings (256x256 / 22 f / 8 steps / seed 0), prompt
        naming NO sound, only `--ref-audio` differing. The reference is a 4 Hz click
        train at 44.1 kHz, so the resampler runs and the reference carries a rhythm
        the prompt cannot express:

        | | baseline | + `--ref-audio` | the reference |
        |---|---|---|---|
        | rms | -55.2 dBFS | **-24.3 dBFS** | -21.6 dBFS |
        | peak | -32.1 dBFS | -4.3 dBFS | -2.7 dBFS |
        | envelope autocorrelation at 4 Hz | **-0.10** | **+0.73** | +0.82 |
        | ...at 3 Hz / 5 Hz | +0.30 / -0.30 | -0.10 / -0.05 | -0.12 / -0.08 |

        The level moving is the weak signal: extra rows perturb the sampler, and the
        baseline is at the known "nothing asked for" floor anyway. The 4 Hz
        periodicity is the strong one -- it is ABSENT in the baseline (the envelope
        is mildly 3 Hz there), appears at +0.73 with the reference, and is negative
        at the neighbouring rates in both the output and the reference. Rhythm at
        the reference's own rate is not something a level change explains.
        (Also confirmed the video is still a video: mid-grey means, ~50 px stddev,
        recognizably the prompt.)
    - **Denoise masks: the MECHANISM is done, the user-facing path is not.**
      A mask never gates the arithmetic; it relabels rows on the time axis. A row
      with mask value `m` runs at sigma `m * sigma_stream`, so `m = 1` is the
      ordinary stream timestep and `m = 0` pins the row at the condition one and the
      model is told it is already clean. `minimax_h3.maskRowValues` +
      `Timesteps.initMasked`, pinned by `tools/gen_minimax_h3_mask.py`, and per-row
      modulation on both the host and CUDA.
      - ⚠️ **The 2x2 patch reduction is `amax`, not a mean.** A patch regenerates if
        ANY of its four latent cells does; a mean makes a half-covered patch
        half-noised, which is a plausible image and not this model's.
      - ⚠️ **The two streams use DIFFERENT sigmas** -- video the sampler's, audio its
        own shifted one. They coincide only at shift parity, which is exactly when a
        parity render could not tell.
      - ⚠️ **The OUTPUT HEADS modulate per row too.** Fixing only the trunk leaves
        them reading the segment's own label, which for a COLLAPSED mask is the
        mask's value and for a per-row one is the unmasked stream's: the same mask
        then renders two different pictures depending on how it happened to be
        expressed. Cost a real debugging pass; found by the isolation below, not by
        the tolerance check, which passed throughout.
      - A mask whose rows all agree COLLAPSES to a segment-level label rather than a
        per-row table, and an all-ones mask reduces to NOTHING. Both are observable:
        they change the set of distinct labels and hence every row index into the
        modulation table.
      - A BINARY mask adds no timestep label of its own (`m = 1` gives the stream's,
        `m = 0` gives exactly the pinned condition one), which is what makes video
        continuation free. A graded one does; the label cap is
        `Timesteps.max_labels` and going past it is refused by name rather than
        quantized, because merging levels would change the picture.
      - Replicate padding up to the latent grid is unobservable for a mask short by
        ONE: with `amax` over 2x2 the replicated cell's source is inside the same
        patch. It only matters from two short.
      - The device takes it through a per-row INDEX BUFFER added to `rms_mod_par`
        and `gated_add` (`rmsModRows` / `gatedAddRows`), not a run-length split: a
        spatial mask alternates row by row, and one launch per run would be
        thousands of launches of a few rows each.
      - **The isolation that matters** is `minimax-h3-cuda-test` with
        `TP_H3_MASK=uniform`: a mask whose rows all agree normally collapses to a
        scalar offset, and `minimax_h3.force_row_labels` keeps the per-row table
        instead. Same labels, same rows, so the two must agree to the last bit --
        and they do, **exactly 0** on host and device. That removes the mask's
        effect and leaves only the plumbing. `binary` / `graded` / `spatial` then
        check device against CPU; those figures are elevated (text rows 0.30-0.44
        against 0.07 unmasked) because a pinned row's modulation is much larger and
        amplifies the existing int8 divergence, which the exact isolation is what
        lets us say rather than assume.
      - **The user-facing half is `--continue-from <dir> --preserve-frames N`**
        (`Session.ContinueFrom` / `Continuation` / `h3EncodeContinuation` /
        `placePreserved`). The source's opening runs through both VAE encoders at
        the TARGET canvas, lands in the sampler's own buffer, and the mask tells the
        trunk those rows are clean. `--continue-from` reads the same
        `frame_NNNNN.png` + `audio.wav` layout `--frames` writes, so a rendered clip
        continues itself.
        - `preserve_frames` is in PIXEL frames and snaps DOWN to the VAE's temporal
          grid (`% 17 == 5`), because that is the only grid its chunking lands on.
          What actually got preserved is reported, not assumed.
        - **Preserved rows are placed ANALYTICALLY at each sigma**, not stepped:
          `x = sigma * noise + (1 - sigma) * ref`, from the ORIGINAL noise draw, both
          before every predict and once more after the last step (nothing re-places
          them after the final Euler move). The reference sets the VELOCITY instead
          (`v = noise - ref`, independent of sigma -- `MiniMaxH3` does not override
          `scale_latent_inpaint`, so it inherits flow noise-scaling); that is the
          same straight line, and placing `x` avoids expressing it in the sampler's
          CARRIED audio space where the velocity picks up the carry's own sigma
          dependence. The reference latent goes through `processLatentIn` for the
          same reason.
        - ⚠️ **An async device upload's host staging buffer must outlive the
          BATCH.** The per-row label tables are uploaded inside the forward's batch,
          and freeing the staging at the end of its own block is a use-after-free
          that read valid memory in three device tests and then hung the device two
          steps into a render -- not at the copy.
        - **Verified by render** at 128x128 / 22 frames / 8 steps, preserving 5
          pixel frames (2 of 7 video latent frames, 8 of 37 audio), against a
          same-seed control with no continuation:

          | vs the source | continuation | control |
          |---|---|---|
          | video frames 0-4 (preserved) | **24.67 dB** | 12.14 dB |
          | video frames 5-21 | 12.72 dB | 12.14 dB |
          | audio, preserved 0.2 s | **66.79 dB** | 47.67 dB |

          The preserved rows end at the encoded source EXACTLY (the final sigma is
          0), so 24.67 dB is the VAE round trip at this size and not sampling error
          -- it is the ceiling, and they are at it. Past the boundary the
          continuation falls to the control's level, i.e. it generates rather than
          copies, and the handover decays over about two frames (25.7 -> 16.9 ->
          13.4 -> 11.1 dB) rather than cutting.
        - ⚠️ A 256x256 continuation does NOT fit a 14 GB cgroup: the VAE encode
          peaks at 4.35 GB on top of the 21 GB checkpoint's page cache and the run
          takes SIGTERM (143) mid-sampling. 128x128 peaks at 1.09 GB and fits. The
          encode pads a short source UP to `clip_length` (17), so preserving 5
          frames costs the same as preserving 17.
11. **GUI**: playback, and the queue/history entries for clips.

## Measured end to end

A 3090, `generate-clip --backend cuda`, as each piece moved to the device:

| shape | steps | sampling | video decode | audio decode | total |
|---|---|---|---|---|---|
| 256x256, 5 f | 1 | CPU everything | — | — | ~13 min |
| 256x256, 5 f | 8 | 14 s | 4.9 s | 37 s | 2 min 21 s |
| **1344x768, 22 f** (native) | 8 | ~7 min | 16.2 s | 16.6 s | ~7.5 min |

The native canvas renders photorealistically with coherent motion. There, sampling
dominates and it is where the FLOPs are: ~272 TFLOP per step at 7138 packed rows, about
half of it attention.

⚠️ **A whole-render wall time here is dominated by paging in the 21 GB checkpoint, not
by the render.** Measured by differencing a 1-step run against a 9-step one at 512x512 /
22 frames (~1850 packed rows), warm: **0.875 s/step**, against a ~40 s floor for load
plus both VAE decodes. So a 24-step 512 clip is ~61 s total and an 8-step one ~47 s --
the two differ by 15%, not by the 4x their step counts suggest. An earlier "~4 min for
512x512 / 8 steps" row in this table was that floor measured cold and is deleted rather
than kept with a caveat. Time a change by differencing step counts, never by one wall
clock.

⚠️ **Shift 12 leaves a large final jump at low step counts.** The schedule packs almost
every step near sigma 1, so 4 steps ends **0.800 -> 0**, 6 steps 0.706 -> 0, 8 steps
0.632 -> 0; 30 (the node default) gives 0.293 -> 0. That jump is what shows up as
under-resolution as the canvas grows.

It is NOT a resolution-independent step floor, which is the mistake to avoid making from
the arithmetic alone: at 256x256 the base model renders a coherent, well-formed subject
at 4 steps with no LoRA at all. Judge a step count at the resolution you intend to
render, and A/B it.

**The 4-step turbo LoRA needs `--shift 1`, not the model default of 12.** Measured at
512x512 / 22 frames / 4 steps, same seed and prompt, looking at the rendered frame:

| config | result |
|---|---|
| no LoRA, 4 steps, shift 12 | soft, mushy legs, under-resolved |
| LoRA, 4 steps, shift 12 | sharper but distorted anatomy |
| LoRA, 4 steps, shift 6 | good, slightly soft |
| LoRA, 4 steps, shift 3 | good, slightly flat |
| **LoRA, 4 steps, shift 1** | **matches the 24-step no-LoRA render** |
| no LoRA, 24 steps, shift 12 | the reference: correct anatomy, sharp fur |

Shift 1 spreads the four sigmas as 1.0 / 0.75 / 0.5 / 0.25, which is what a distilled
4-step model was trained against; shift 12 puts three of the four above 0.92 and leaves
a 0.8 -> 0 jump the LoRA was never distilled to cover. Nothing in the LoRA's metadata
says this, so it had to be swept. There is no per-LoRA schedule default yet: pass
`--shift 1` by hand.

⚠️ **Moving the video shift alone rescales the AUDIO**, which is why there are two
knobs. The reference has independent `shift_video` (12) and `shift_audio` (3), and the
sampler's audio carry is their RATIO (`minimax_h3.Shifts.carryScale`). A `--shift 1`
with the audio shift left at 3 takes that ratio from 4 to 1/3, and the audio comes out
hard-clipped instead of quiet. `--shift-audio` exists so a caller can hold the ratio:
`--shift 1 --shift-audio 0.25` keeps it at 4. Both are recorded in the MP4's
`parameters` tag (`appendClipParams`), because a reader re-renders from it.

Caveat worth keeping: this file is a **ref2v** turbo LoRA and the renders above are t2v
with no reference, so it is being used outside what it was trained for and may do better
still on the reference path.

## Prompting: the audio layer is IN the prompt

⚠️ **The audio level is conditioned by the prompt text, and a prompt that describes
no sound renders near-silence.** This cost a wrong conclusion: a whole "the audio is
30-50 dB down, something upstream is wrong" investigation, resolved by changing the
prompt. Measured at the nominal shifts, same seed, 256x256 / 22 frames / 8 steps:

| prompt | peak | rms |
|---|---|---|
| `a red fox running through deep snow, cinematic` | 2.2% FS | -52.4 dBFS |
| structured, with an explicit `overall_soundscape:` and `non_diegetic_music:` | 100% FS | -17.7 dBFS |

35 dB from the text alone, neither clipping, and the loud one's envelope rises and
decays with a 17.8 dB crest factor (an impact, not flat noise).

The model wants a STRUCTURED prompt, not a caption. The sections it is trained on:

```
subject_definitions:   <Subject N> / <Scene N> definitions, reused by reference below
summary:               one-paragraph pitch, with the shot count and duration
retention_analysis:    per-subject, what a reference image preserves (ref2va only)
detailed_description:  [Shot N] blocks with timecodes, camera moves, speed ramps
overall_soundscape:    diegetic SFX, chronological, with intensity words
non_diegetic_music:    score: instrumentation, mood, dynamics
```

Nothing in the engine parses these; `text_encoders/minimax.py` tokenizes the prompt
as raw text. They matter because the model was trained on them. The reference's own
prompt-writing guidance lives in `comfy_extras/nodes_textgen.py`, which is a
captioner helper rather than part of the model, and it says the load-bearing part
outright: *"Audio layer: describe complete soundscape... align audio intensity with
action tempo"*, and *"if the request implies sound, describe it plausibly"*.

Two consequences for measurement:

- **The audio latent's rms is a bad proxy for the audio level.** It moves 0.327 ->
  0.395 across that 35 dB swing. `TP_AUDIO_MAG=1` is for "is the vocoder's INPUT
  sane", nothing more.
- **Every visual judgement in this file was made on the fox prompt**, which
  under-uses the format. The shift and step-count conclusions above are single-prompt
  single-seed eyeballs and are worth re-running on a structured prompt before they
  are treated as settled.

## Development shapes

Do not develop against the default canvas. `length = 5` at 256x256 gives `latent_t = 2`,
`audio_t = 8`, a 8 x 8 patch grid: **128 video rows + 16 audio rows + text**. Small
enough for a CPU forward and for fixtures with teeth, and it exercises every code path
the full shape does except the temporal chunking in the VAE decode.

Verify a fixture has teeth by breaking the implementation deliberately and confirming it
fails. A correct port of the wrong convention passes every test.

## Silent wrong answers

Every one of these is a plausible-looking render or a clean-sounding waveform when got
wrong. This list is what reading the reference bought; add to it.

**The DiT**

- **The output is negated.** `forward` returns `[-video_out, -audio_out]`. Both streams.
- **adaLN has three modality tags** (video 0, text 1, audio 2) and the projection
  reshapes to `[M * modalities, expand * hidden]`, so the mod row is `t_row * 3 + tag`.
  Interleaving timestep and tag the other way is silent.
- **The text span is not uniform.** Vision-pad tokens inside it carry tag 0 (video), not
  tag 1, so the span splits into tag runs. The tagged region widens by one on each side
  of a vision block, to cover `<|vision_start|>` / `<|vision_end|>`.
- **adaLN curve lookup**: `t` clamps to `[0, 1]`, scales by `grid - 1`, and the lower
  index clamps to `grid - 2` so `t = 1.0` lands on the last interval instead of reading
  past the table.
- **RoPE is partial split-half.** `[S, 3] -> [S, 48] -> cat(half, half) -> [S, 96]`, and
  the rotation table takes only the first half. Head dim is 128, so 96 dims rotate and 32
  pass through untouched.
- **The position grid is area-normalized floats, not indices.** Per axis,
  `linspace((1 - ratio) / 2, (1 + ratio) / 2, dim / patch, endpoint=False) * 32` with
  `ratio = dim / sqrt(h * w)`. float64 throughout, cast to f32 only at rope time.
- **The video time axis is non-uniform**: span `FRAME_RESCALE * FRAME_PER_TOKEN[k % 5]`
  with an exclusive cumsum, so the first token of every group of 5 spans 1 frame and the
  rest span 4.
- **Audio rows are channel-major** (`ch0 t0..T-1`, then `ch1 t0..T-1`), their `w`
  coordinate pinned to the *extremes* of the video frame's `w` grid (low for channel 0,
  high for channel 1), `h` fixed at 0.
- **`pack_audio` is `latent[0].permute(1, 2, 0)`**: `[B, 32, 2, T] -> [2 * T, 32]`.
- **The patchify order is the einsum `nctrhpwq->nthwcrpq`**, and the unpatchify is its
  inverse. Layout permutations are rms-preserving; pin each direction with its own test.
- **The audio carry**: the sampler holds `(sigma_a / sigma_v) * x_audio`; `forward` undoes
  it and converts the velocity back as
  `(1 - scale) * (x * carry) + (1 + (scale - 1) * sigma_a) * out`. Getting this wrong
  leaves the video looking correct and the audio subtly wrong.
- **Condition noise augmentation restarts the same RNG stream for every condition**
  (the generator is seeded inside the loop). Video uses `seed`, audio uses `seed + 1`.
- **Refs pack between text and the targets, and push the target timeline out.** The
  cursor starts at `text_len` and advances by each ref's span before the targets are laid
  down.

**The video VAE**

- **Encoder GroupNorm is per-frame**, time folded into batch. A normal 3D GroupNorm is a
  silent wrong answer.
- **`CausalConv3d` is reflect-padded spatially and zero-padded front-only temporally**,
  and the single-frame case truncates the temporal taps rather than convolving zero frames.
- **The ViT3D decoder appends 4 register tokens plus 1 zero token**, gives them rope ids
  of 0, and drops them after `proj_out`. `mask_token` is unused at inference.
- **Its attention needs `nan_to_num(0)`** after the attention call; the reference does
  this explicitly.
- **Its token ids are `arange(0.5, dim) / dim * 2 - 1`** in `[-1, 1]`, with rope angle
  scale 2*pi and base 100.
- **Latent normalization is per-channel (mean, std)**, not a scalar scaling factor, and
  the values live in the checkpoint metadata under `minimax_h3_video_vae`.
- **`_finalize_pixels` undoes ImageNet normalization and clamps to `[0, 1]`.** The VAE
  wrapper's output processing is identity, so there is no extra `* 2 - 1`.
- **Temporal chunking on decode** (`clip_length` 17, `token_drop` 3, chunk 5, overlap 2,
  `frame_pre_padding` 3, `frame_overlap` 5) is what determines the output frame count.
  The arithmetic is intricate; port `_decode_temporal_frame_plan` rather than re-deriving it.

**The audio VAE**

- **Snake is `x + sin^2(alpha * x) / beta`**, and alpha/beta are stored in **log scale**
  for the decoder's `SnakeBeta` but **linear** for the encoder's `Snake1d`, where alpha
  also serves as beta.
- **The anti-aliased activation is upsample x2 -> snake -> downsample x2** with
  kaiser-sinc filters and replicate padding, not a plain pointwise activation.
- **BigVGAN's final output is clamped to `[-1, 1]`** with no tanh and no final bias.

**The text encoder**

- **The conditioning is the unnormalized hidden state after layer 50.** No final norm.
- **Per-token modality tags ride alongside the embeddings** and drive the DiT's adaLN
  tag runs. They are not recoverable from the embeddings.

**The LoRA sidecar**

- **`scale = strength * alpha / A.rows`**, divided by the rows of `lora_A` as the FILE
  stores them. A fused qkv factor has A `[3r, in]` and an alpha already multiplied by 3,
  so the per-block reading and the whole-tensor reading agree only at the stored count.
  Using the per-block rank on the fused alpha is exactly a 3x error.
- **`delta = B @ A`, with A `[rank, in]` and B `[out, rank]`.** Whenever `rank == in`
  the transpose is representable and multiplies fine.
- **The fused qkv's B is block diagonal.** Splitting it is what makes it cheap; dropping
  an off-diagonal entry that was not actually zero is silent, which is why the detector
  tests the data and declines on the first nonzero.
- **`swi_glu_mapping` in the LoRA's metadata is already applied.** `[value;gate] ->
  [gate;value]` describes what was done to the file, not what a reader must do. Applying
  it again swaps the halves.
- **A sidecar must reach every GEMM call site.** One fast path that skips it renders
  plausibly and wrongly with no error. `Lin` makes the pairing structural; the weight is
  reachable only as `.w`.
- **The sidecar works in the UNROTATED activation space.** `opI8Prep` reads the f32
  activation rather than rewriting it, which is what lets the base int8 GEMM and the
  bf16 sidecar share one source buffer. Feeding the sidecar the rotated int8 staging
  buffer would be finite and wrong.
- **Attach before sizing any device workspace.** The sidecar scratch is sized from what
  is attached; a LoRA attached afterwards has nowhere to run.
