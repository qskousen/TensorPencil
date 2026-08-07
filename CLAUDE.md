# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Ground Rules

- Never run `git add` or `git commit` unless directly requested.
- Don't bring up "this code is uncommitted"; don't worry about commits or checkpoints or anything like that.
- `zig build` produces no output on success; any output indicates a warning or error.
- **NEVER run a built binary from the cache directly (e.g. `./.zig-cache/o/*/<exe>` or any hardcoded/globbed cache path) — always launch through the `zig build <step> -- <args>` command.** The cache holds *multiple stale binaries* from earlier builds; a glob or copied path silently runs an old one that predates your edits, producing bogus results (this has burned us more than once — e.g. a "no speedup" benchmark that was actually running pre-change code). `zig build` recompiles and runs the *current* source every time. This applies to benchmarks (`embed-bench`, `ggml-bench`, …), `run`, `run-llm`, `run-gui`, and any other exe.
- `zig build test` is likewise silent when everything passes. Tests must NOT print diagnostics on success — use `errdefer std.debug.print(...)` before the assert so values only print on failure. Any stderr from a *passing* test makes the runner print a misleading red `failed command:` line (see ZIG.md); if `zig build test --summary all` says `test success`, nothing failed — don't investigate that line.
- **Read `ZIG.md` before doing any work.** It documents Zig 0.16.0 breaking changes relevant to this codebase. When you encounter and resolve a new 0.16.0 change, add it to `ZIG.md`.
- We want the code to be clean, with clear seperation of concerns, modular, and testable.
- If there is ambiguity in a request, don't guess or assume; ask for clarification.
- When adding a new feature or fixing a bug, add unit / integration tests as appropriate.
- Make sure all tests are still passing after working on something. If they aren't, fix it - even if the test was previously broken.
- **Default to `zig build test` (fast, ~15s CPU unit suite). Do NOT run `zig build test -Dintegration` unless you truly need it** — it runs the GPU device tests and real-model inference tests and takes ~11 minutes. Reach for `-Dintegration` only when your change touches GPU kernels / device code or the real-model LLM/parity paths, and even then prefer narrowing with `-Dtest-filter="<substring>"`. See the Commands section for the full split.
- Remember that the best code is often the simplest code. Tend towards simple solutions where possible, that will work for all the edge cases.
- If the user asks for something that may cause issues, push back and get confirmation before doing it.
- If you see existing code that may cause issues or is Band-Aid patch code, call it out and suggest a fix.
- This is a research project. There's no risk to trying big complicated work. We want to try unusual things. Be bold and adventerous.
- However, we do want code to be structured, organized, and allow for generalization as much as possible without sacrificing performance.
- **Cross-platform code; don't lock ourselves into Linux-only.** Even where a subsystem currently only runs on Linux (e.g. the CUDA/NVIDIA backend), reach for portable std APIs (`std.Io` futex/mutex/sleep, `std.posix`, `std.Thread`) over raw Linux syscalls (`std.os.linux.*`) unless there's a real reason none of them fit — so a future macOS/Windows port isn't blocked by avoidable platform lock-in. If you must go platform-specific, gate it behind a comptime `builtin.os.tag` branch with a portable fallback and call it out.
- After adding a new kernel feature like relo, supporting a new dtype like bf16 or qk_6 for a backend, or anything similar, check BACKEND.md and update it to reflect the current state.
- Performance is CRITICAL, and we need to do what it takes to get there - don't skip out and do something easier if the hard work is what is needed.
- **A negative/limiting conclusion requires a receipt.** Before claiming an optimization "isn't worth it," "won't help," "can't be done cleanly," or "is too fragile/expensive," you must have an ISOLATION measurement that removes exactly the component in question (e.g. disable the op and re-time) — not a proxy and not an assumption. State whether each claim is measured or assumed.
- **A result that contradicts a strong prior means the measurement is suspect, not the prior.** (A 3090 being "flat" on batched matmuls is physically implausible → verify the harness before concluding — stale binaries, contention, wrong build.)
- **Name shortcuts explicitly and default to the robust option.** If an approach trades robustness for effort, say so, state the robust alternative and its real cost, and lead with the robust one — don't silently pick the easy path. A real tooling limit gets a clean workaround that does the full job, never a fragile hack or a reduced-scope "halfway."

## Project

TensorPencil is a diffusion inference engine (text-to-image) plus an LLM inference engine (`tp-llm`, see `LLM_PLAN.md`) written in Zig, targeting **Zig 0.16.0** (`minimum_zig_version` in build.zig.zon).
The central parts are also exported as a library.

## Commands

- `zig build` — build the executables (installs to `zig-out/bin/TensorPencil` and `zig-out/bin/tp-llm`)
- `zig build run -- <args>` — build and run the diffusion CLI with arguments
- `zig build run-llm -- <args>` — build and run `tp-llm`, the LLM inference CLI (see `LLM_PLAN.md`)
- `zig build test` — run the **fast CPU unit suite** (~15s; both module test binaries). The slow integration tests are gated OFF by default (see below).
- `zig build test -Dintegration` — run **everything**, including the GPU (CUDA/Vulkan) device tests and the real-model LLM/parity tests that load multi-GB checkpoints and run inference in Debug (~11 min). Needs a device and the `models/` checkpoints; individual tests still self-skip when their specific device/file is absent.
- `zig build test --fuzz` — run fuzz tests (`std.testing.fuzz`)
- `zig build -Doptimize=ReleaseFast` — optimized build (important for benchmarking inference; Debug is very slow for numeric code)

The `-Dintegration` gate lives in `src/test_gate.zig` (a `build_options.integration` flag): GPU `init` fails in test builds when it's off so device tests self-skip, and heavy real-model tests call `test_gate.requireModelFile`/`requireIntegration`. Gate a new slow test the same way; keep fast CPU unit tests ungated.

To run a single test, filter with the build option (reuses the build cache):
- `zig build test -Dtest-filter="<substring>"` (add `-Dintegration` for a gated test), or `zig test src/root.zig --test-filter "<substring>"` for a standalone compile. Note the **`=` is required** for the build option: the space-separated form `-Dtest-filter "x"` fails with *"Expected -Dtest-filter to be a string, but received a flag"*, and passing the option twice fails with *"received a list"* — only one filter substring per run.

## Architecture

Two-module layout wired in `build.zig`:

- **`src/root.zig`** — root of the public `TensorPencil` module (created with `b.addModule`, importable by consumers). All engine code (tensors, model/weights loading, schedulers/samplers, text encoder, UNet/DiT, VAE decode, image output) belongs under this module; anything public must be re-exported from `root.zig`, since consumers can only reach declarations visible from the module root.
- **`src/main.zig`** — root of the unnamed executable module; the CLI. It imports the library via `@import("TensorPencil")` and should stay a thin argument-parsing/driver layer over the library.

Tests live in both modules; `zig build test` builds and runs two separate test executables (one per module) in parallel.

## The pipeline stages

`pipeline.Session.generate` is composed from four public stages, and a caller can
drive them directly instead — that is what img2img, inpainting, custom samplers,
latent upscaling, drift curves and per-step measurement all need:

```zig
var cond = try sess.encode(gpa, prompt, .{});          // text  -> conditioning
const sigmas = try pipeline.schedule(gpa, steps, shift); // steps -> sigma schedule
var den = try sess.denoiser(gpa, cond, null, 1.0, lat_h, lat_w, sigmas);
try den.predict(gpa, v, x, sigmas[i], null);           // one denoiser forward
var img = try sess.decode(x, lat_h, lat_w, .{}, null);  // latent -> RGB8
```

- **`Denoiser` exists because `predict` cannot be a free function on the GPU
  backends.** The text fusion, rope table, timestep vectors and activation
  workspace are built once per image; rebuilding them per step would be both slow
  and a behaviour change. It is bound to one resolution and *borrows* its
  conditionings, which must outlive it; the `sigmas` argument is a cache (the
  Vulkan session precomputes a timestep vector per entry and recomputes for an
  off-list sigma), not a constraint on what `predict` accepts. `Session.predict` is the one-shot form
  (init + predict + deinit): free on CPU, wasteful on a GPU backend, and exactly
  what a single-forward measurement wants.
- **`Session.decode` does not modify the caller's latent** (it denormalizes onto a
  copy), unlike the in-place version that lived inside `generate` — a caller
  holding the latent for a drift curve would otherwise have it silently rescaled.
  There is a test for it.
- ⚠️ **Both families share ONE VAE-decode ladder (`Session.decodePlanar`), and the SD
  family was outside it until 2026-08-03 — which hard-failed every render whose decode
  did not fit.** The ladder is whole-image → free VRAM and retry → GPU-tiled → CPU-tiled,
  and `recoverableDecodeErr` already named the exact symptom this produced ("the hand-PTX
  path can surface a post-OOM stream fault as `CudaError`"); the SD arm just returned
  before reaching any of it. Reported as
  `cuMemcpyHtoD failed: CUDA_ERROR_ILLEGAL_ADDRESS` → `error: image generation failed:
  CudaError`, because `upload` is a blocking legacy-stream copy and so the first call to
  *observe* a kernel that already faulted. At 1024×1536 the SD decoder wants **3 × 1.5 GiB**
  of activations, so any resident LLM or foreign process is enough.
  - The family-specific half is a small adapter (`KreaVae` / `SdVae`) supplying the
    per-backend tile context, the peak-VRAM estimate and the tile geometry. **A
    whole-image decode is just a single tile covering the whole latent**, which is what
    collapses the two arms into one ladder.
  - ⚠️ **SD needs HALF krea2's tile (64² latent, not 128²), and a wrong tile size makes
    tiling pointless rather than broken.** The SD decoder's widest activation is 256
    channels at *full* image resolution, so a 128² latent tile is 3 × 1 GiB — barely under
    a 1024×1536 whole-image decode. 64² is also what ComfyUI's tiled SD decode defaults to.
  - `vae_tiled.decode` derives the latent channel count from `z.len / (zh·zw)` instead of
    krea2's 16, and its fast test now runs both 16 and 4 channels.
  - `sdTile` owns the planar ↔ channel-last transposition on both sides of the SD
    decoders, so `vae_tiled` and `decode` stay planar-only. It is one function for the
    reason the layout section below gives: the wrong transposition is rms-preserving, so
    no magnitude check can see it. A fast test pins each direction separately.
  - The SD arm also now `setMemTag(.vae)`s, so the GUI's VRAM meter stops attributing a
    multi-GiB decode to the untracked "ovh" segment (it read `vae=0MB` before).
  - `--vae-decode auto|whole|gpu_tiled|cpu_tiled` exposes the override on the CLI; it was
    reachable only through the library and the GUI before.
- ⚠️ **The invariant to protect: `generate` composed from the stages must be
  bit-identical to `Session.generate`.** The gated test
  `"generate composed from the public stages is bit-identical to Session.generate"`
  builds the same image both ways at 128² on CPU and compares the RGB bytes, and
  also asserts one-shot `Session.predict` equals the loop's first forward. When the
  split was made, the three backends (CPU, Vulkan, zig-cuda incl. CFG) each
  produced a **byte-identical PNG** to the pre-refactor binary. Preserve that: if a
  change makes the stages and `generate` disagree, every measurement taken through
  the stages describes a model nobody renders with.
- VRAM tagging is part of each stage (`encode` → `.te`, `denoiser` → `.latent` then
  `.dit`, `decode` → `.vae`), so the GUI meter stays meaningful for a caller that
  drives the stages itself. The per-image CUDA *pinning policy* stays inside
  `generate`, since it is about a queue of images rather than one stage.

## Checkpoint containers: the DiT loads from GGUF too

`dit.DiT.load` takes a `weights.WeightStore`, not a `*const SafeTensors`, and
`pipeline.DitContainer` opens either format by extension (`.gguf` → GGUF). Before
this the DiT was safetensors-only, which meant **half of ggufy's output — every
ggml block quant — could not be run or measured at all**, and every image-level
comparison in the quantization programme was an int4-safetensors arm.

Two conventions had to be honoured to make a real GGUF load:

- ⚠️ **`comfy.gguf.orig_shape.<name>` must be restored, and `gguf.zig` now does it
  at parse time.** ComfyUI-GGUF's converters (and ggufy's `applyShapeFix`) reshape
  any tensor whose contiguous dim is not a multiple of 256 to `(n/256, 256)` so
  ggml's blocks tile it, recording the true shape in that KV. krea2's patch embed
  therefore arrives as `[1536, 256]` instead of `[6144, 64]`, and without the
  restore every shape check downstream fails on a perfectly well-formed file.
  - ⚠️ **The restore used to be skipped for block-quantized dtypes, and that made
    every ggufy SD-family GGUF unloadable (fixed 2026-08-02).** The reasoning was
    that "the criterion that triggers the fix — a contiguous dim not divisible by
    256 — is the same one that makes k-quantization impossible, so ComfyUI only ever
    applies it to tensors it then leaves unquantized". True of ComfyUI's converter;
    **false of ggufy's**, which blocks over a tensor's *flat* element count, so it
    reshapes first and k-quantizes happily. Measured on a ggufy SD1.5 q4_k file:
    **166 of 686 tensors, 72.7% of the parameters** — every convolution, whose
    contiguous dim is `kw` (1 or 3). Symptom: `sd_unet: time_embed.0.weight has
    shape { 1600, 256 }, expected [1280, 320]`. So the SD family's whole GGUF path
    was unmeasurable — the same gap this section closed for krea2, still open here
    because nothing had tried it.
  - The shape is now restored for those too and the tensor flagged
    `TensorInfo.flat_blocks`. The **values** are fine — the fix is a pure regrouping,
    so flat row-major order is preserved, and 256-wide storage rows hold exactly one
    block each, meaning flat and per-row blocking coincide with or without an
    imatrix. The **layout** is not: a logical row of 320 is not a whole number of
    q4_k blocks, so `matmul.Weight.init`'s row-aligned assumption fails. Consumers
    needing row-aligned blocks therefore materialize (`sd_unet.mat` and `clip_text`
    already had that branch; `dit.zig`'s `mat` refuses loudly, since krea2 cannot hit
    it — its only shape-fixed tensor, `first.weight`, is `keys_hiprec` and so
    unquantized).
  - ⚠️ **Named cost: materializing gives back the memory quantization saved.** SD1.5
    q4_k is ~0.5 GB on disk and ~3.4 GB f32 resident (a divergence arm runs ~10 GB
    RSS). The robust alternative is a block GEMM that understands flat blocking on
    all four backends — a real kernel project, deliberately not undertaken here. This
    path is functionally complete and purely additive: these files previously
    hard-errored, so nothing that loaded before changes behaviour.
  - ⚠️ **Related structural limit, worth stating right here:** no GPU DiT runs
    block-quantized weights at all (`dit.gpuLinKindSupported` is
    `{i8, i4, bf16, f8_e4m3}`; `sd_unet_cuda`'s GEMM switch is `{f32, f16, bf16}`).
    So a GGUF arm must be measured on `-b cpu`, and there is **no weight-only GPU
    render path for any architecture** — k-quants are weight-only but CPU-only, while
    int4/int8 ConvRot run on GPU and quantize the activation in place (W4A4).
- GGUF stores dims reversed; `gguf.zig` already un-reverses them. ggufy writes bare
  diffusion tensor names (`blocks.0.attn.wq.weight`), which `canonicalName` passes
  through untouched — only `blk.*` and the three LLM specials are rewritten.
- `DitContainer.open` picks the reader by **magic, not extension**. Guessing wrong
  sends a GGUF to the safetensors parser, which reports `InvalidHeader` — an error
  that says nothing about what happened. (Cost an initially misdiagnosed measurement
  failure: the real cause there was a *stale binary* linked against the pre-change
  TP, and the useless error is what made it look like a bad file.)

**Verified end to end:** a mixed q4_k/q5_k krea2 GGUF loads (28 blocks, per-layer
dtypes preserved, no convrot metadata) and renders — 31 s at 256²/3 steps on CPU,
**19.2 dB PSNR / 0.775 SSIM** against its own bf16 base. `mat()` also now names the
tensor and both shapes when a shape check fails; a bare `ShapeMismatch` across 230
weights is not actionable.

⚠️ **A GGUF DiT runs on CPU only, and that is now enforced.** The block quants have
no GPU GEMM path here: `dit_cuda` always refused them, but `dit_gpu` (Vulkan)
recognizes int8/int4/bf16 and treats **everything else as raw fp8-e4m3** — so the day
`pipeline` learned to open a GGUF, Vulkan started rendering a **blank white image with
no error**. Both forwards now gate on `dit.gpuLinKindSupported` and return
`error.UnsupportedCheckpoint`; a fast-suite test pins the classification. If a GPU
block-quant GEMM ever lands, that predicate is the one place to widen.

**What the CPU path actually computes**, since "runs a GGUF" is ambiguous: at
`m >= ops.matmul.small_m_max` (16) `matmulPacked` **dequantizes the weights into f32
panels** and multiplies in f32 — weight-only quantization, W4A16, no integer
arithmetic. Below 16 it takes ggml's `vec_dot` GEMV, which also quantizes the
*activations* (Q8_K for k-quants) — a genuinely quantized kernel, but a DiT forward at
any real resolution is hundreds of tokens and never enters that regime. So a
block-quant DiT measures format loss, not kernel loss.

## Weight overlays: substituting one tensor without rewriting a checkpoint

`weights.Overlay` is a third `WeightStore` arm — a base store plus a
name→`TensorView` patch map from caller memory — and `Session.replaceDit` swaps a
live session's DiT for one loaded from any store, keeping the tokenizer, text
encoder, VAE and backend untouched.

Both exist for ggufy's per-layer attribution measurement (quantize exactly ONE
tensor, run the whole model, see what moved), where the alternatives are both fatal
to it: writing a modified 26 GB checkpoint per data point, or rebuilding the
`Session` per point — which would **re-encode the conditioning**, putting a
difference in the *inputs* of a measurement whose whole claim is that the inputs are
identical. They are equally useful for merged LoRA/delta weights, per-layer dtype
experiments, and a GUI hot-swap.

Four things the implementation commits to, three of them traps:

- ⚠️ **`mapping()` returns null for an overlay**, not the base's mapping. A patched
  tensor's bytes are outside it, and that accessor exists so a consumer can turn a
  view into a mapping-relative offset for direct GPU streaming — which would read
  the *unpatched* bytes for exactly the tensors an experiment changed. Losing the
  streaming fast path is a slowdown; that would be a silent wrong answer.
- ⚠️ **`replaceDit` evicts the device weight caches.** They are keyed by *host
  pointer*, and the outgoing DiT's arena is freed — a later allocation landing on one
  of those addresses would score a stale cache hit against another tensor's device
  copy.
- **An overlay may substitute, never extend.** `put` requires the name in the base
  and checks `bytes.len` against shape × dtype, so `names()`/`count()` stay exactly
  the base's and an f32 buffer handed in as bf16 is rejected rather than read as
  noise. The dtype *may* change (f32 in place of bf16 is the normal case — it holds a
  dequantized round-trip exactly); the shape may not.
- **Lifetimes:** patch bytes must outlive the model, and the `Overlay` must stay at a
  stable address while a store refers to it. Loaders keep views, not copies.

Pinned by unit tests plus one on a real krea2 checkpoint that asserts a one-tensor
substitution moves *nothing else* — every other weight pointer-identical to an
unpatched load. (`first`/`last.linear` are materialized to f32 at load, so those two
compare by value.)

## The SD family (SD1.5 and SDXL)

Why it exists: ggufy's whole activation-aware quantization programme has measured **one**
architecture (krea2), and its central finding — that per-tensor damage spans only 2.3x around the
median, so precision routing cannot beat uniform allocation — is a statement about a stack of 28
identical transformer blocks. A UNet is the most structurally different thing available (conv-dominated,
four channel widths, multi-resolution), and SD1.5/SDXL are the two architectures whose hand-built
`sensitivities/*.json` files ship today and have never been validated. SD1.5 is also 860M params, so its
whole measurement ladder runs in minutes rather than krea2's hours.

**Reference fixtures come from `tools/gen_sd15_fixtures.py`** — `transformers.CLIPTextModel`,
`diffusers.UNet2DConditionModel`, `diffusers.AutoencoderKL` — in two tiers: pure-torch op references
(`src/ops/assets/sd15_op_fixtures.json`, ungated) and real-checkpoint parity
(`src/models/assets/sd15_ref.safetensors`, ~1.5 MB, `-Dintegration`, tied to one checkpoint by sha256).
Full tensors where small, per-stage `(mean, l2, max)` triples where not, so a mismatch localizes to a
block instead of failing at the end.

| piece | state |
|---|---|
| `ops.norm.groupNorm` | ✅ matches `F.group_norm` |
| `ops.act.geluQuick` / `geluErf` (ungated forms) | ✅ |
| `models.clip_text` (CLIP-L) | ✅ matches `CLIPTextModel` on a real checkpoint |
| `models.sd_unet` (SD1.5 config) | ✅ matches `UNet2DConditionModel`, rel L2 < 1e-3 |
| `models.sd_vae` (AutoencoderKL decoder) | ✅ matches `AutoencoderKL.decode`, rel L2 < 2e-3 |
| `sampler.sd*` (discrete-eps schedule) | ✅ matches `EulerDiscreteScheduler` (ladder, 4- and 10-step sigmas, timesteps) |
| **SDXL** — CLIP-L + CLIP-G, UNet, `y`, VAE | ✅ all match **ComfyUI**; see the SDXL section below |

⚠️ **The layout convention is the one that actually bit, and it is rms-preserving.** The sampler works in
planar `[c][h][w]` (krea2's DiT and both VAEs do, and `generate`/`decode` are written to it) while
`sd_unet.forward` works in channel-last `[h*w][c]`. Feeding one to the other moves every value's *position*
and none of its magnitude, so per-step eps/x norms matched the reference to a fraction of a percent while
the render became a periodic streak pattern. Two things hid it for a long bisection: the UNet parity tests
**transpose explicitly** before calling `forward`, so they validated the UNet and not the pipeline's
convention; and a dump-and-compare against diffusers **interpreted the dumped buffer the same wrong way**,
so both sides scrambled identically and "agreed" to 1e-5. The tell was that magnitudes agreed while the
image did not — that signature means a permutation, not an arithmetic error. `Denoiser.predictSd` owns the
transposition now, and a fast test pins the round trip *and* the fact that the wrong layout has the same
norm. **Verified end to end: 70.6 dB against a ComfyUI reference render** (512², 20 steps, CFG 7.5, same
seed; max per-channel difference 2/255). That figure was 50.9 dB against a *diffusers* render until the two
sampling conventions below were fixed — see "⚠️ Two conventions where ComfyUI and diffusers disagree".

⚠️ **A parity test is only as good as the reference's *configuration*, and this cost a visibly bad
image.** The first schedule matched `EulerDiscreteScheduler` exactly — configured with
`timestep_spacing="leading"` (the old PNDM discretization), which starts a 4-step run at index 751,
**sigma 4.12 instead of 14.615**. So the sampler was told the latent was only moderately noisy while
`scaleInitialNoise` had scaled it as pure noise: no global structure formed and the render came out as
noise-textured mush. The implementation was a *correct* port of the wrong convention, and every test
passed. `linspace` (999 → 0, `steps_offset = 0`) is what k-diffusion Euler uses for SD1.5 and what
ComfyUI/A1111 sample with; `sampler.sdTrainIndex` owns that choice and documents it.

⚠️ **Three conventions where ComfyUI and diffusers disagree, and the SD family follows ComfyUI.** The
first two were found 2026-08-01 by rendering SDXL against a ComfyUI render of the same seed; the third on
2026-08-03 while landing DPM++ 2M SDE. None produces an error, none changes the *composition*, and all
three are worth many dB:

| | diffusers | ComfyUI (**what we do**) | measured |
|---|---|---|---|
| initial noise scale | `max_sigma` (for `linspace`/`trailing` spacing) | `sqrt(1 + sigma_max²)` | SDXL 22.0 → **30.9 dB** |
| UNet timestep | the fractional index | **nearest trained index** (`argmin` in log-sigma) | SDXL 30.9 → **56.1 dB** |
| **schedule interpolation** | lerp **sigma** at a fractional index | lerp **log-sigma**, then `exp` | SD1.5 SDE 19.9 → **bit-exact schedule** |

Cumulative for the first two: SDXL 22.0 → 56.1 dB (RMSE 0.4/255), SD1.5 50.9 (vs diffusers) → **70.6 dB**
(vs ComfyUI, max diff 2/255). The second is the bigger one and the less obvious: the sinusoidal timestep
embedding is continuous, so a fractional index is perfectly well-defined — it is just not what the ecosystem
conditions on, and at 8 steps the schedule's indices sit ~0.3 off an integer in *every* step.
`sampler.sdScaleInitialNoise` and `sampler.sdModelTimestep` own these, document the disagreement, and a fast
test pins both so neither can revert to the diffusers form. `Session.scaleInitialNoise` dispatches the first
by family (krea2's flow matching must keep multiplying by exactly 1.0).

⚠️ **The third one is the interesting case, because it is a bug whose visibility depends on the sampler.**
Both implementations read the same (bit-identical) 1000-rung beta ladder at the same `linspace(999, 0, steps)`
indices; they differ only in the space the fractional index is interpolated in. `sampler.interpLadder` lerped
sigma (diffusers' `EulerDiscreteScheduler`); ComfyUI's `ModelSamplingDiscrete.sigma` lerps `log_sigmas` and
exponentiates. Worth up to **4.4e-5** mid-schedule, which is:

- **invisible to the fixture**, which compares against *diffusers* at the 2e-4 the log/exp round trip forces;
- **invisible to Euler** — 53.83 dB before the fix, 53.59 after, i.e. unchanged, because the residual there
  is the UNet's own ~1e-4 and Euler's trajectory is smooth in sigma;
- **fatal to any SDE sampler**, because `brownian.zig` quantises the sigma axis to 1e-6 and uses it as a
  **tree key**. 4.4e-5 is 44 cells, so every step drew unrelated noise. Receipt: ComfyUI rendered against
  itself with **one** sigma of twenty nudged across **one** cell is **28.9 dB**; nudged by 1 ulp *within* a
  cell it is 69.4 dB. The quantity is chaotic in its last digits and smooth in its value.

`sdSchedule`/`interpLadder`/`sdTrainIndex` now compute in **f32, matching the reference's rounding rather
than bounding it** — the opposite of `timestepEmbedding`'s f64 policy, and deliberately so: a quantised key
has to agree digit for digit, where a numeric quantity only has to agree closely. That includes reproducing
`torch.linspace`'s f32 `step` and halfway split for the index, and the fact that `exp(log(x))` is not the
identity in f32. Result: the 4- and 10-step schedules are **bit-exact** to ComfyUI, 19/21 and 29/31 sigmas
land in the right `round6` cell at 20 and 30 steps (the residual is one ulp of torch's vs Zig's `expf`).
`test "the SD schedule matches ComfyUI's `normal` scheduler, not diffusers'"` pins it at 2.4e-7 relative
*and* asserts the cell-agreement fraction, since the tight bound alone does not describe what the SDE
sampler consumes.

**The generalizable lesson, and it is the fourth time this file has had to record a version of it:** matching
a reference implementation exactly still leaves the *choice of reference* unvalidated. The schedule bug above
was a correct port of the wrong `timestep_spacing`; the noise scale and timestep were correct ports of
diffusers where the compatibility target is ComfyUI; the interpolation space was a correct port of diffusers
that *no existing test or render could see* until a sampler came along that consumed the value differently.
A render comparison against the actual target is what caught all four, and nothing in a fixture suite would
have. **Corollary worth keeping:** "this tolerance is loose enough for the current consumer" is a claim with
an expiry date.
| `core.clip_tokenizer` (CLIP BPE) | ✅ matches `CLIPTokenizer` on 10 adversarial cases |
| `Session` family generalization | ✅ `pipeline.Family`, both families behind one stage API |

### One `Session`, three families

`Session.models` is a `union(Family)` over `Krea2Models` (three checkpoints) and `SdModels` (normally
one) — **the same struct for both `.sd15` and `.sdxl`**, since they differ only in `Config`s and in
SDXL's second text tower, so `denoiser`, `predict`, `decode` and teardown are shared verbatim rather
than duplicated. Bind `Session.sd()` once and no further family test is needed.

The stage API is unchanged — `encode` → `denoiser` → `predict` → `decode` dispatch internally, so
**every existing caller, including ggufy's whole measurement ladder, works on all three families**
(ggufy needed *no* changes for SDXL). What made that possible is that the Euler step consumes the same
quantity either way: krea2's velocity and SD's eps are both the trajectory derivative.

New family-aware surface, because these are the things a caller genuinely cannot decide for itself:
`Session.family()`, `.schedule(gpa, steps, shift)` (krea2 reads `shift`; SD's ladder comes from the
training betas and ignores it), `.latentChannels()` (16 vs 4), `.denoiserStore()` and
`.replaceDenoiser()` — the last two replacing `dit_st`/`replaceDit`, since "the denoiser's container" is
now the family-neutral concept the per-tensor attribution arm needs.

- **The family is detected from the denoiser's own tensor names** (`detectFamily`), never from a flag: a
  mistyped `--arch` in front of a measurement is exactly the input error the harness exists to exclude.
  Both spellings resolve — a full single-file checkpoint keeps the `model.diffusion_model.` prefix,
  ggufy's model-only GGUF strips it — and an unknown architecture is an error rather than a default.
### Container style is orthogonal to architecture

⚠️ **Joined-vs-split is NOT a property of the family.** Any architecture ships either as one bundled
checkpoint (denoiser + text encoder + VAE in a single file, each under its own prefix) or as separate
files, and the same model is distributed both ways. An earlier version of this code tied the two together
— krea2 always split, SD allowed to be joined — which is simply wrong.

`resolveComponent` decides per component (`denoiser` / `conditioner` / `decoder`) with the precedence the
CLI promises:

1. **An explicitly given `--text-encoder` / `--vae` wins**, even when the primary checkpoint also carries
   that component. Overriding a bundled component is the entire reason to pass the flag.
2. Otherwise the primary checkpoint's own copy, under any of its known prefix spellings.
3. Otherwise a *defaulted* side path (`Options`' krea2 files).
4. Nowhere → `error.ComponentNotInCheckpoint`, rather than a confusing failure deep in a weight load.

⚠️ **A defaulted path is not a request, and conflating the two broke a joined SD1.5 checkpoint**: the
resolver dutifully opened krea2's qwen3 encoder (the `Options` default), looked for CLIP's tensors in it,
and reported `ComponentNotInCheckpoint`. Hence `Options.explicit_text_encoder` / `explicit_vae`, which a
CLI sets only when the flag was actually present — and hence a defaulted side path that does not exist on
disk is not an error, while an explicit one that cannot be opened still fails loudly.

⚠️ **`generate`'s latent length is `latentChannels()`, not 16.** Hardcoding krea2's count ran SD's
4-channel UNet correctly for four steps and then failed in `decode` with `LatentSizeMismatch` — the
sampling loop only ever handles whole latents, so the mismatch surfaced as far from its cause as possible.

**`weights.Prefixed` is what makes that cost nothing per loader.** It is a fourth `WeightStore` arm — a
base store plus a prefix — so a loader is always handed a store in which *its* component sits at the root,
exactly as if it had come from a dedicated file. No loader takes a prefix parameter, no caller needs to
know which spelling a file used, and it generalizes to every future architecture. (`qwen3.TextEncoder.load`
and `wan_vae.Decoder.load` now take a `WeightStore` instead of a `*const SafeTensors` for this reason.)
⚠️ Its `names()` returns the **stripped** names, built once at construction: a consumer that enumerates a
prefixed view and then looks those names up — the activation-capture sanity gate does exactly that — would
otherwise miss on every one.

The concrete case this exists for: ggufy writes a GGUF holding **only** the UNet, so a *quantized* SD arm
is measured with the UNet from the GGUF and CLIP + VAE from the original checkpoint, while an unquantized
arm reads all three out of one file.
- **The SD family runs on every backend** as of 2026-08-01 (`sd_unet{_gpu,_cuda}.zig`,
  `sd_vae{_gpu,_cuda}.zig`) — see "### The SD family on the GPU" below and BACKEND.md §2A.
  It was refused on any GPU backend before that, and the refusal was the right default while
  it lasted: the failure mode of pretending otherwise is on the record, since when `pipeline`
  learned to open a GGUF, Vulkan rendered a blank white image with no error.
- ⚠️ **`sampler.scaleInitialNoise` is applied unconditionally and is a no-op for krea2.** Flow matching
  starts at sigma = 1, so the multiply is by exactly 1.0 (bit-identical, and the function early-returns);
  SD's ladder starts near **14.6**, and skipping it hands the UNet a latent 15× too small — it denoises
  something that was never noisy and produces a washed image with no error anywhere.
- ⚠️ **`ops.conv.Conv2d` now carries a `tag`.** A convolution here *is* an im2col GEMM, so
  `ops.matmul.probe` always fired on it — but untagged, meaning an activation capture of a UNet would
  have recorded only its attention and feed-forward linears and silently omitted the convolutions holding
  most of its parameters. Same class of partial coverage as the Vulkan capture that recorded 39 of 263
  layers and passed its own sanity gate. The im2col GEMM's columns are `ci·kh·kw`, which is exactly how
  ggufy quantizes a rank-4 conv weight — so a per-column imatrix lines up without reinterpretation.
- **A named `Models` union, not an inline one:** an anonymous `union(Family)` in the field makes the
  compiler derive its name from the first field's type and then report a dependency loop.
- ⚠️ **The live per-step PREVIEW is a latent *format*, and it was krea2-only until 2026-08-03.** Both
  halves of it hardcoded krea2's 16 channels — `downsampleLatent`'s channel loop and the
  `wan_vae.latentPreviewInto` call — so the first previewed SD render in the GUI read four planes past the
  end of a 4-channel latent and **panicked mid-generation** (`index out of bounds: index 0, len 0`), on
  SD1.5 and SDXL alike. `Session.latentPreviewInto` dispatches by family now, alongside
  `latentChannels()`/`scaleInitialNoise`, and `downsampleLatent` takes the channel count. Two things that
  are not interchangeable even within the SD family: SD1.5 and SDXL have **different** latent2rgb
  matrices, and SDXL carries a **bias** where SD1.5's is zero (from ComfyUI's `latent_formats`) — the
  wrong one is not a crash, just a preview with plausible structure and wrong colours.
  - The taesd-quality preview stays **krea2-only on purpose**: TAEHV (`taew2_1`) is a 16-channel Wan
    approx-VAE and cannot decode an SD latent at all. The SD equivalents (TAESD / TAESDXL) are a
    different, unimplemented decoder, so a taesd request from the GUI degrades to latent2rgb through the
    existing `method == 2 and taehv_dec == null` path. Loading it anyway is what produced the panic.

**The SD family's sampling differs from krea2's in three ways that all have to line up**
(`sampler.zig`'s discrete-eps section): sigma comes from a discrete beta ladder rather than a formula, the
model is conditioned on a **timestep index** so a chosen sigma must be inverted back to a (fractional)
index, and the input is pre-scaled by `1/sqrt(sigma²+1)`. The *step* is the same Euler, because for
eps-prediction the trajectory derivative **is** eps — which is what lets ggufy's teacher-forced level 2
compare both families without knowing which it is looking at.

⚠️ **Two module-boundary facts worth knowing before adding fixtures.** `tp_core` is rooted at
`src/core/core.zig`, so it cannot `@embedFile` anything under `src/ops/` or `src/models/` — each module
owns its own fixtures (hence `src/core/assets/clip_tokenizer/fixtures.json` separate from
`src/ops/assets/sd15_op_fixtures.json`), and it cannot import `test_gate` either, so a `tp_core` fixture
test self-skips by catching the open error instead.

⚠️ **The VAE's GroupNorm epsilon is 1e-6, the UNet's is 1e-5**, and `decoder.up.N` is indexed from the
**outermost** level while the forward runs `up.3 … up.0`. Reading the list forwards gives a decoder that
runs and inverts the channel ramp.

**The UNet is the first convolutional, multi-resolution model here.** Activations stay channel-last
`[h*w][c]` (so a 1×1 convolution *is* a GEMM over pixels and `ops.conv` handles the 3×3s), the skip stack
is LIFO and includes the stem's output (12 skips for 12 output blocks), and the `Workspace` sizes its
ping-pong buffers for the widest stage once per resolution rather than per step. Two naming traps cost a
cycle each: `ops.conv.conv2dBanded` takes an explicit band size (`conv2d` is the defaulted form), and a
`res`/`spatial` helper that builds an already-prefixed base name and passes it back through the prefixing
loader produces `model.diffusion_model.model.diffusion_model.…`, which surfaces as a *missing tensor*
rather than as the naming bug it is.

⚠️ **`timestepEmbedding` computes in f64 on purpose.** diffusers computes it in f32, and at `i = 0` the
argument is the timestep itself (~1000), so a 1e-7 relative slip becomes ~1e-4 in `cos`. Being *more*
accurate than the reference bounds the disagreement by the reference's own rounding instead of stacking
two errors — hence a 2e-4 fixture tolerance rather than 1e-6.

⚠️ **`GroupNorm`'s statistics are not per-position**, unlike every other norm here: it reduces over
`h·w·(channels/groups)` values at once, and uses the **biased** variance (divide by `n`) to match torch.
With 640 values per group the biased/unbiased difference is ~0.08% — too small to look like a bug.

⚠️ **Three checkpoint-reality traps, all hit while landing CLIP-L**, and each cost a debugging cycle:

- **An SD1.5 merge in the wild stores its CLIP linears as `f64`.** `convertToF32` now reads f64 (ComfyUI
  casts them too), and `ops.matmul.supportsDType` lets a *loader* decide up front whether to materialize
  a weight to f32 instead of discovering it from inside the first forward. Integer dtypes stay refused:
  silently floating `position_ids` is never right.
- **`.arena = arena` in a struct literal copies the arena's state before later fields allocate into it**,
  so anything allocated by a *later* field initializer leaks — invisible except to the test allocator.
  Build every field into a local first, then construct. (`dit.zig` already does; `clip_text` had to
  learn.)
- **`CLIPTokenizer(vocab_file=…, merges_file=…)` silently produces meaningless ids** — without the
  directory's `tokenizer_config.json` / `special_tokens_map.json` there is no bos/eos/pad, so every word
  becomes the unknown token. The hidden-state comparison *passed* on those ids, because both sides were
  fed the same garbage; only the pooled-output lookup (no eos to find) caught it. Fixture content is part
  of a fixture's correctness.

### SDXL

Same three files as SD1.5 (`sd_unet`, `sd_vae`, `clip_text`) plus a second text tower — no new model
files, because the SD1.5 work left the right seams. What changed:

| | SD1.5 | SDXL |
|---|---|---|
| levels | 4 (`{1,2,4,4}`) | 3 (`{1,2,4}`) |
| attention | levels 0–2 | levels **1–2** (none at the outermost) |
| transformer depth | 1 everywhere | `{_, 2, 10}` per level |
| heads | `num_heads = 8` (head_dim grows) | `num_head_channels = 64` (**head count** grows) |
| `proj_in`/`proj_out` | 1x1 conv `[c,c,1,1]` | `nn.Linear` `[c,c]` |
| conditioning | CLIP-L, 768 | CLIP-L ++ CLIP-G, **2048** |
| extra | — | `y` = pooled ++ 6x256 size sinusoids → `label_emb` → `+= emb` |
| VAE | scale 0.18215 | scale **0.13025**, otherwise identical |
| skips / output blocks | 12 | 9 |

**Reference is `tools/gen_sdxl_fixtures.py`, and it drives ComfyUI, not diffusers** — a deliberate
departure from the SD1.5 generator. ComfyUI is the compatibility target (renders here are verified
pixel-identical to it), and more importantly it is what *settles SDXL's conventions*, none of which are
derivable from the checkpoint. Hand-converting the state dict inside the generator would have made the
fixture and the code under test share an assumption. Run it with ComfyUI's **`nvenv`**, not the
ai-toolkit venv.

⚠️ **Five conventions, each of which yields a plausible image when wrong.** This is the whole content
of the SDXL work; the arithmetic was already there.

1. **CLIP-G is stored in OpenCLIP naming with a fused `attn.in_proj_weight`** while CLIP-L, *in the same
   file*, is a `transformers` CLIPTextModel. `clip_text.Config.Naming` carries which; the `[3h, h]` fused
   matrix's three row blocks are q, k, v in that order, taken as zero-copy views.
2. **Both towers condition on the PENULTIMATE hidden state with the final LayerNorm SKIPPED**
   (`layer_idx = -2`, `layer_norm_hidden_state = False`). Using the final state is the "clip skip"
   difference and shifts style everywhere.
3. **CLIP-L pads with EOS, CLIP-G pads with 0** (`"!"`), so the prompt is tokenized **twice**
   (`clip_tokenizer.encodePadded`). The pad slots are inside the 77-token window the UNet cross-attends
   to and a causal tower gives them distinct hidden states, so this is conditioning, not filler.
4. **Pooled comes from CLIP-G only**, from its *final-LayerNormed* last hidden state at the first EOS,
   through `text_projection` — stored in `x @ T` orientation, i.e. the transpose of an `nn.Linear`
   weight (`projectPooled` does the transpose explicitly; the reference implementations flip it at load).
5. **`y` is pooled ++ six 256-wide sinusoids of (height, width, crop_h, crop_w, target_h, target_w)** —
   h before w, LDM's order. Any permutation is a valid 2816-vector that SDXL absorbs as a differently
   *composed* image. `sd_unet.admVector` + `MicroCond`, pinned by a fixture rather than reasoned about.
   The micro-conditioning is built in `Session.denoiser`, not `encode`, because it carries the image
   size; defaults are original == target == the render, no crop, which is what ComfyUI and diffusers
   both use (ComfyUI derives them from the latent shape).

⚠️ **`detectFamily` must test SDXL before SD1.5, and by `label_emb`.** Both are LDM UNets with the same
`input_blocks.0.0` stem, so the stem cannot distinguish them; reading it first loads every SDXL
checkpoint as SD1.5 and fails at the fourth level's missing weights instead of saying what it is. The
detection test's SDXL cases deliberately carry **both** tensors, so they pin the *order* — a case with
`label_emb` alone would pass either way.

⚠️ **Two ComfyUI-as-a-library traps, both silent.** Worth knowing before writing another generator
against it:

- **`comfy.options.enable_args_parsing()` must be called before importing `comfy.cli_args`**, or argv is
  parsed as `[]` and *every* flag silently defaults — the "CPU fp32" reference was running on the GPU
  with xformers, whose only symptom was an unrelated-looking `NotImplementedError` deep in attention.
  The generator asserts the flags took, precisely because there is otherwise no diagnostic.
- **Forward hooks do not fire where you expect.** `text_model.embeddings` is bypassed (ComfyUI computes
  token embeddings outside it so textual-inversion vectors can be spliced in, passing `embeds=`), and
  `input_blocks[i]` is never *called* (the UNet hands each `TimestepEmbedSequential` to
  `forward_timestep_embed`, which iterates the children and dispatches on their type). Capture the
  encoder's *input* for the former and patch that function for the latter. Getting this wrong silently
  produced a fixture with **zero** stage rows, not an error.

**Verified end to end: 56.1 dB against a ComfyUI reference render** (512², 8 steps, CFG 7.5, same seed;
RMSE 0.4/255, 0.02% of pixels off by more than 8/255) — after the two sampling conventions above, which
this comparison is what found. 8.8 s/step on 8 CPU cores in ReleaseFast, against ComfyUI's 25 s/step at
CPU fp32.

**Measured, all 17 SDXL tests green** (`zig build test -Dintegration -Dtest-filter=SDXL`, 7 min):
CLIP-L and CLIP-G both match ComfyUI at rel L2 < 3e-3 (including the fused-QKV split, checked at the
embedding output and after layer 0 before 32 layers can smear a mistake), the concatenated 2048-wide
context matches row by row *and* half by half (which pins the concatenation order), pooled matches, `y`
matches `encode_adm` to 1e-6, the UNet matches at 16² and 64² at rel L2 < 1e-3 with no per-phase
structure, and the VAE matches at both sizes.

⚠️ **`Workspace.init` now takes `ctx_seq`**, and this was a latent bug, not a tidy-up. Cross-attention
writes `ctx_seq` rows of keys/values into `k`/`v`, which at a small latent **exceeds** the position count
(a 16² latent gives 64 positions at the innermost level against 77 context rows) — the old
positions-only sizing was sufficient only by accident, via an allocation ~10x larger than needed. Sizing
the attention scratch per *attending* level at its own resolution also drops a 1024² SDXL render's
feed-forward buffer from 671 MB to 84 MB, which matters because SDXL's outermost level does not attend
at all.

`--text-encoder-2` overrides CLIP-G for a split-file checkpoint (a bundled one carries it, and
`conditioner2` resolves independently of `conditioner` so `--text-encoder` overrides only the first tower).
⚠️ **That split path is unexercised** — every SDXL checkpoint here is bundled, so it is built and reviewed
but not measured; asking krea2 or SD1.5 for `conditioner2` is `error.NoSuchComponent`.

### Long prompts: attention weights and 77-token chunking

⚠️ **`clip_tokenizer.encode`/`encodePadded` truncate at ONE 77-token window, and until
2026-08-03 that was the whole prompt path — so every real prompt rendered a different
image than ComfyUI's.** This was reported as "same prompt, negative, seed, steps, cfg,
sampler, scheduler — what gives?", with two visibly different SDXL renders. It is worth
recording precisely because nothing failed: no error, no warning, and a composition
close enough to the reference that it read as a numerics problem.

Measured on the reported prompt (115 CLIP tokens): ComfyUI built **154** conditioning
rows, this engine built **77** and silently dropped everything from `(lens flare:0.75)`
through `newest` — which is to say the lens flare (visibly present in one render and
absent in the other) and the entire `masterpiece, best quality, very aesthetic, high
contrast, vibrant` block, hence one saturated image and one flat one.

Two independent gaps, and the second makes the first bite earlier than the prompt's
length suggests:

1. **No chunking.** ComfyUI packs into as many whole 77-token windows as it takes and
   concatenates along the sequence axis. `encodeWeighted` now does; `Cond.seq` was
   already threaded through every backend, and both GPU UNets already sized
   cross-attention `@max(n, ctx_seq)`, so nothing below the encoder needed changing.
2. **No emphasis parser.** `(shiny skin:1.1)` tokenized *literally* — nine content
   slots spent on punctuation that ComfyUI strips, at weight 1.0. About 25 of the 75
   usable slots in that prompt were syntax.

⚠️ **The weight is an interpolation away from the EMPTY PROMPT, not a multiply**, and
not A1111's mean-renormalized form either:

```
z[j] = (z[j] - z_empty[j]) * w + z_empty[j]
```

`z_empty` is the same tower run on `[BOS] [EOS] pad…` at the same capture layer, indexed
by the same position `j`. So `w = 0` means "whatever this slot says with no prompt at
all" rather than a zero vector. All three candidate forms produce a plausible image; a
plain multiply misses ComfyUI by **3.1e-2** where the correct form lands at **2.2e-6** —
a factor of 14,000, which is why this gets a fixture rather than an argument.
`clip_text.applyWeights` owns it, and the two GPU arms call that same function rather
than reimplementing it.

**Four conventions that are not derivable, all verified against ComfyUI's own code:**

| | |
|---|---|
| `BREAK` | **not honoured** — it tokenizes as the literal word `break`. A1111 pads to the next chunk on it; ComfyUI has no such rule and the split is purely length-driven. |
| bare `(a)` | multiplies the weight by 1.1, cumulatively (`((a))` = 1.21) |
| explicit `(a:1.5)` | **replaces** it absolutely — `((a:1.5))` is 1.5, not 1.65. The inner value wins over every enclosing multiplier. |
| short segments | a tokenized segment under `max_word_length` (8) is pushed **whole** into the next chunk rather than split across the boundary; a longer one is split. |

Also: the weight product accumulates in **f64** and narrows once (Python's `1.1*1.1` is
1.2100000000000002, not f32's 1.2100001); a chunk's EOS goes **immediately after its
content**, with padding after it, never before; and `pooled` comes from **chunk 0 only**
and is never weighted — taking it from the last chunk would condition SDXL's `y` on the
quality tags alone.

**Pinned in two tiers, because the ids can be right while the weighting is wrong:**

- `tools/gen_clip_prompt_fixtures.py` → `fixtures_weighted.json` (ungated, `tp_core`):
  31 adversarial prompts × 2 paddings against ComfyUI's own `SDTokenizer`, exact on ids
  *and* weights. Verified to have teeth — setting `max_word_length` to 999 or dropping
  the 1.1 multiplier each fails 6 cases.
- `tools/gen_clip_cond_ref.py` → `clip_cond_ref.safetensors` (gated): ComfyUI's
  `CLIPTextEncode` output on two real checkpoints. Agreement **1.2e-6–2.7e-5** rel L2
  across 14 prompt × tower comparisons.

**End to end, like-for-like against ComfyUI** (SDXL, 1024×1536, 35 steps, CFG 5,
`dpmpp_2m_sde_heun` + `karras`, same seed): **12.85 dB → 35.56 dB** (SSIM 0.584 → 0.982),
and **38.30 dB** with `euler` at the same settings — which is the isolation showing the
residual is the SDE amplification the sampler section documents, not the conditioning.

⚠️ **One trap in taking that measurement**: the reference PNG had been produced with
`VAEDecodeTiled` (tile 512), and ComfyUI's own tiled decode differs from its whole-image
decode by **28.25 dB**. Comparing against it caps any result near 28 dB and looks like a
real defect in this engine. Re-render the reference with the decode you are actually
comparing to.

### Two prompt dialects: ComfyUI and AUTOMATIC1111

`--prompt-syntax comfy|a1111` (`Options.prompt_syntax`, plus `--emphasis
original|no_norm|ignore`). Landed 2026-08-03 after the chunking work above, because the
same prompt text is **not the same prompt** in the two ecosystems — five things differ, and
every one of them changes the image:

| | ComfyUI (`clip_tokenizer.encodeWeighted`) | A1111 (`core/prompt_a1111.zig`) |
|---|---|---|
| `(x:w)` | **replaces** the weight | **MULTIPLIES** the enclosed range |
| `[x]` | literal text | **1/1.1 de-emphasis** |
| `BREAK` | the literal word `break` | **forces a chunk boundary** |
| 75-token boundary | hard cut | **backtracks to the last comma** (<=20 back) |
| application | `(z - z_empty)*w + z_empty`, per position | `z*w`, then **one global rescale** restoring `z.mean()` |

⚠️ **Row 1 is the counter-intuitive one.** Because `:w` multiplies, `(((house:1.3))` is
`1.3 x 1.1 x 1.1 = 1.573` — the two unclosed parens still apply, since A1111 flushes open
brackets at end of input rather than discarding them. Row 5 is the other trap: A1111's
rescale is a single scalar over the whole chunk, so **emphasising one tag moves every other
token in that chunk**, including the ones at weight 1.0. `no_norm` skips it and is what
upstream itself recommends for SDXL.

**Plus per-step scheduling**, which no other prompt path here has: `[a:b:when]` swaps a
span mid-render and `[a|b]` alternates it every step. ⚠️ **`when` is a fraction of `steps`
only when the literal contains a `.`** — `[b:3]` is step 3, `[b:.5]` is halfway, so `[b:1]`
and `[b:1.0]` are different schedules.

**The reference is A1111's `modules/prompt_parser.py` EXECUTED, not re-derived.** It
imports standalone (only `re` + `lark`), so `tools/gen_a1111_prompt_fixtures.py` runs its
real Lark grammar. ⚠️ **A1111 is AGPL-3.0 — the generator fetches it at run time and it is
never vendored**; the fixture records the upstream sha256.

⚠️ **Upstream parses with Earley and this is recursive descent, so equivalence is
MEASURED**: 283 schedule cases agree exactly (43 hand-picked, every upstream doctest
included, plus a 240-case seeded random corpus over the stressing alphabet at 4/10/20
steps). It rests on reproducing upstream's *failure* behaviour, since on any parse error it
discards scheduling and renders verbatim. Two rules make real prompts fail, both in
`unschedulable`:

- **A bare `|` is legal ONLY directly inside a MATCHED `[...]`.** Top level, inside `(...)`,
  or inside an unbalanced bracket all fail — so **NovelAI-style `{a|b}` prompts silently
  lose all scheduling in A1111**, and reproducing that is the difference between matching
  the reference and being "reasonable".
- **An alternation option may not contain a top-level `:`** (`[a|b:0.5]` fails,
  `[a|(b:0.5)]` parses — there the `:` is inside parens).

⚠️ **One upstream doctest is STALE and fails upstream too** (`a [{b|d{:.5] c`, 1 of its
24). Pin what the code does, not what the docstring says — `doctest.testmod` on the
reference is how you find that out.

⚠️ **Two fixture groups initially had NO TEETH**, found by deliberately breaking the
implementation and seeing nothing fail. Both are now covered and the second is guarded:

1. `:w` multiply-vs-replace is indistinguishable unless a case nests an explicit weight
   *inside another* (`((a:1.2):1.5)` -> 1.8 vs 1.5): replace-then-multiply-later coincides
   with multiply-then-multiply whenever the inner weight is applied first.
2. `comma_padding_backtrack` **never fires on an alternating `word, word,` prompt** — the
   boundary always lands ON the comma. It needs unpunctuated words carrying the boundary
   past a comma. The generator now **asserts** at least one case distinguishes the setting
   on from off and refuses to emit a corpus that does not.

**How a step-dependent prompt reaches the model**, and what it deliberately does not
disturb:

- `Cond.sched` holds the extra conditionings plus a per-step index, while
  `data`/`seq`/`pooled` **stay entry 0** — so every existing caller, including ggufy's
  measurement ladder and the composed-stages invariant, is untouched. `Denoiser.predict`
  keeps its signature and means entry 0; `predictAt(step)` is the scheduled form.
- ⚠️ **Dedup by text is load-bearing, not an optimization.** `[a|b]` contributes a schedule
  entry for EVERY step, so 35 steps is 35 entries over 2 distinct texts; encoding per entry
  would mean 33 wasted tower forwards and 33 wasted device sessions.
- ⚠️ **`seq` can differ between entries**, so the shared UNet workspace is sized over every
  entry of both branches, not just entry 0 — cross-attention writes `ctx_seq` rows into it.
- ⚠️ **The sigma schedule is now computed BEFORE the text encode**, because the prompt
  schedule is indexed by step and the count is `sigmas.len - 1` (not `opts.steps` —
  `ddim_uniform` and `beta` return a different number). The hoist is safe: `scheduleWith`
  is pure and touches no RNG, and the gated bit-identical-stages test still passes.
- ⚠️ **The PNG `parameters` block now records `Prompt syntax:` (and `Emphasis:` under
  a1111).** A reader re-renders from that block and the same text means a different image
  per dialect; A1111's own format has no such field because A1111 has only one dialect.
  Same reasoning that stopped `Sampler` and `Schedule type` being hardcoded.
- ⚠️ **Prompt weights that the loaded text encoder cannot apply are WARNED about and the
  prompt encoded verbatim — not refused, and not silently dropped.** krea2 is the case:
  qwen3 tap states have no fixed token window, so emphasis has nowhere to apply.
  - **This was `error.PromptSyntaxUnsupportedForFamily` and it went through two wrong
    versions before this one.** First blanket (reported as "why does krea2 report
    `PromptSyntaxUnsupportedForFamily`?"), which failed even a *plain* prompt — the dialect
    is a **persistent** setting, so a global choice made for an SDXL render broke every later
    krea2 render, on prompts where the two dialects are byte-identical. Then narrowed to
    weighted prompts, which was still stricter than any live tool.
  - **What settled it: emphasis weighting is a CLIP-only feature in every implementation.**
    Checked against **SD.Next** (`dev`) rather than A1111, which never had to decide, having
    only ever had CLIP. `processing_prompt.set_prompt` gates its emphasis parser behind an
    allow-list of `StableDiffusion*` / `StableCascade` / `Flux`; everything else —
    Qwen-Image, Wan, Lumina, Flux2, krea2's shape — falls to `prompt_attention = 'fixed'` and
    the raw string reaches the pipeline unparsed. **SD3 is the clincher**, having both kinds of
    encoder at once: the CLIP towers go through the weighted provider while the T5 tower is
    `_get_t5_prompt_embeds(prompt=…)` on the raw string, the two then concatenated. Flux is
    *on* the allow-list and still encodes unweighted ("clip is only used for the pooled
    embeds"). Chroma and HiDream sit **commented out** of the list: tried, then disabled.
  - The warning is what keeps the house rule intact — nothing is silent here, where SD.Next
    only `debug_log`s its own fallback. **Verbatim rather than stripped** is deliberate: it
    keeps `.comfy` and `.a1111` byte-identical on such a model (verified: identical md5 on a
    krea2 render), and `--emphasis ignore` is the explicit way to parse the syntax away.
- ⚠️ **The capability is read off the ENCODER (`TextEncoder.supports_prompt_weights`), not
  from a list of architectures**, and SD.Next is the argument for that too: its allow-list
  matches **substrings of pipeline class names**, which needs a `'Flux2' not in cls` guard so
  `Flux` does not swallow a different architecture, and needs editing for every new pipeline —
  silently giving the wrong answer until someone does. `pipeline.supportsPromptWeights(fam)`
  reads the declaration on the encoder actually loaded, so it cannot drift from the real
  capability, and a newly added encoder must state its answer or **fail to compile**. The three
  properties that decide it are documented on `clip_text.TextEncoder`: a fixed token window,
  rows the denoiser cross-attends to 1:1, and an empty-prompt encode of matching shape to
  interpolate against.
- **Per-step scheduling works on every family**, including krea2, and is a genuine feature the
  ComfyUI dialect cannot express (there `[…]` is literal text): `Cond.sched` is family-neutral
  and `encodeText` runs once per resolved text. ⚠️ Which is why the weight check runs on the
  **schedule-resolved** text — `[cat:dog:0.5]` reaches `encodeText` as `cat`/`dog`, and
  checking the raw string would read a bare `[…]` as de-emphasis. The test goes through
  `planSchedule` for that reason.

### Reproducing an A1111 render: the RNG, and two more sampling conventions

`--compat comfy|a1111` (`Options.compat`, plus `--rng cpu|nv`, `--sgm-noise-mult on|off`,
`--quantize-t on|off` as per-knob overrides). Landed 2026-08-04, after the prompt-dialect
work above, because **matching A1111's prompt syntax is not enough to reproduce an A1111
image** — the same seed was still a different picture. Three more things differ, all of
them *sampling* rather than text, and every one was hardwired here to ComfyUI's choice:

| | A1111 default | ComfyUI (was unconditional here) | cost |
|---|---|---|---|
| `randn_source` | **`GPU`** — NVIDIA Philox | torch CPU MT19937 | **an unrelated image** |
| `sgm_noise_multiplier` | **`False`** — `x·σ₀` | `x·sqrt(1+σ₀²)` | ~9 dB |
| `enable_quantization` | **`False`** — fractional σ→t | nearest *trained* index | ~25 dB |

⚠️ **The RNG is categorically worse than the other two, and it is the one nothing here
could have caught.** The second and third perturb a trajectory; the first replaces the
starting point with an unrelated draw. Upstream's own description of the option is "changes
seeds drastically". The other two are already documented from the *other* direction — the
`sdScaleInitialNoise` and `sdModelTimestep` doc comments name **diffusers** as the side
that disagrees with ComfyUI, and A1111 happens to sit on diffusers' side of both. So this
is the third ecosystem to land on the same two forks in the road, which is the argument for
`Compat` being a named axis rather than three loose flags.

**`core/philox_rng.zig`** is the port: Philox4x32-10 + Box-Muller, with
`core/noise.zig` as the two-engine selector (`torch_cpu` | `nv_philox`) that both consumers
dispatch through. Reference is A1111's own `modules/rng_philox.py` — a pure-numpy CPU
imitation of CUDA's generator, which is what its third `randn_source` setting ("NV")
selects — so the target is executable and **no GPU-side RNG work was needed**.
`tools/gen_philox_fixtures.py` runs it (AGPL: fetched at generation time, never vendored,
sha256 recorded) and the port is **bit-exact** on the block cipher and on every normal,
including an FNV-1a hash over whole latent-sized draws.

⚠️ **It must apply to the Brownian tree as well, and that half is invisible on the default
sampler.** The k-diffusion commit A1111 pins (`ab527a9a`) constructs
`torchsde.BrownianTree(t0, w0, t1, entropy=s)` on `w0 = zeros_like(x)[0]` — the latent's own
**CUDA** tensor — with no `cpu` kwarg anywhere, where ComfyUI's fork forces the tree to the
CPU. So under A1111 *every* SDE node is a Philox draw too. Wiring only the initial latent
would have made `euler` reproduce perfectly while every SDE render stayed wrong, with
nothing failing — the same shape as the Vulkan capture that recorded 39 of 263 layers and
passed its own gate. `SdeStepper.Options.noise_src` → `brownian.NoiseSampler.src`.

⚠️ **`rng_philox` is faithful to real CUDA only below a HARDWARE-DEPENDENT element count,
and the numpy code's shape actively misleads about why.** ATen's kernel
(`DistributionTemplates.h`) gives each thread one `curand_normal4` and spreads its four
normals across a grid stride, so element `i` is *not* obviously thread `i`'s first normal.
It is, because the launch is `grid.x = min(SM_count · (maxThreadsPerSM/256),
ceil(numel/256))`: below the cap there is one thread per element and only `rand.x` is ever
consumed — which is exactly `counter[2] = i`, sine only. ATen's own comment says so ("for
common tensor sizes, we end up using rand.x from each thread"). On a 3090 the cap is
**125,952 elements** — above a 1024×1536 latent's 98,304, below a 1536²'s 147,456. Past it
the tail comes from the cosine partners and the noise genuinely differs. `Generator.threads`
reproduces that regime and a test pins the equivalence below the cap / divergence above it;
**nothing in the CLI sets it**, deliberately, because reproducing a *foreign* A1111 render
would require knowing which GPU drew it. A1111's own "NV" setting has the identical limit.

⚠️ **The arithmetic is numpy's f64, not curand's f32**, because `uint32 * float32` promotes
to **float64** under NumPy's rules: `box_muller` runs in f64 with f32-rounded constants and
narrows once, where curand's device path is f32 throughout and uses the `__sincosf`
intrinsic. So this is bit-exact against `randn_source="NV"` and ~1e-6 relative against
`"GPU"` — two orders below the 1.3e-4 model-level disagreement the render comparison already
carries, and the f64 side is the one with an executable reference.

**Also recorded in the PNG `parameters` block**, on the same reasoning that stopped
`Sampler` and `Schedule type` being hardcoded: `Compat: A1111` when not ComfyUI, plus
`RNG:`/`SGM noise multiplier:`/`Quantize timesteps:` **only when overridden away from that
compat's own defaults** — so an ordinary ComfyUI render's block is byte-for-byte what it was
before this existed. `RNG` keeps A1111's own field name and its `CPU`/`NV` spellings.

**Verified end to end, with the control in the same run** (SD1.5 dreamshaper_8, 512²,
8 steps, CFG 7.5, euler + `normal`, seed 12345, CPU fp32 both sides). A1111 is not installed
here, so the reference is **ComfyUI with exactly those three conventions substituted** — real
`rng_philox` noise, `Sampler.max_denoise` forced False (which is precisely what turns
ComfyUI's `noise_scaling` into `noise · sigma`), and `model_sampling.timestep` returning
k-diffusion's unquantized fractional index:

| | vs its own reference | the two arms against each other |
|---|---|---|
| `--compat comfy` | **64.05 dB**, SSIM 0.9999 | |
| `--compat a1111` | **66.42 dB**, SSIM 1.0000, max 5/255 | **8.55 dB**, SSIM 0.289 |

Three things that table says. The a1111 arm lands *slightly better* than the control, so all
three conventions are wired right rather than partially right. The 8.55 dB between the arms is
**identical on both engines** (ComfyUI's own two arms differ by the same 8.55 dB / 0.289), which
is the cross-check that the divergence is the conventions and not this implementation. And 8.55
dB is what "an unrelated image" looks like numerically — for scale, the *prompt dialect*
difference is 19.6 dB and the SDE-amplification residual is 19–28 dB.

⚠️ **The reference is a simulation of the target, not the target** — it shares this reading of
A1111's source, so it validates the wiring (through an independent UNet, VAE and sampler loop)
and not the reading. What is independently pinned is the piece the reading could most easily
have got wrong: the noise itself, bit-exact against upstream's own `rng_philox`.

**Still owed, and it needs a device:** `rng_philox` == `torch.randn(device='cuda')` is argued
from ATen's and curand's source above but **not measured** — the GPU was fully occupied when
this landed. One `torch.randn((4,64,64), device='cuda')` against `Generator(seed).randn` settles
it; expect agreement to ~1e-6, not bit-exact, for the f64-vs-f32 reason above.

**Two things deliberately NOT done**, since both would be guesses: A1111's `subseed` /
`seed_resize_from` / `eta_noise_seed_delta` (all no-ops at their defaults, which were
checked), and the capped-thread regime above.

### The SD family on the GPU

Landed 2026-08-01: `sd_unet_gpu` / `sd_unet_cuda` and `sd_vae_gpu` / `sd_vae_cuda`, so SD1.5
and SDXL run on **vulkan, zig-cuda and cuda** as well as the CPU. BACKEND.md §2A holds the
support grid and the measured table (25-41x end to end on `cuda`); what follows is what
a change here has to keep true.

**Both stages map the way `vae_gpu` maps the Wan decoder** — activations stay tight
channel-last `[h*w][c]` f32, a 3x3 convolution is a banded `im2col_sd` plus a GEMM, a 1x1
convolution and an `nn.Linear` are that GEMM with no patch step. What a UNet adds over a VAE
decoder is GroupNorm, a SpatialTransformer, and the LIFO skip stack; the SD VAE decoder then
reuses the UNet's convolution and GroupNorm helpers rather than copying them (`convInto`,
`groupNormInto`), so the two SD stages cannot drift apart.

**The CLIP towers moved to the GPU on 2026-08-03** (`clip_text_gpu` / `clip_text_cuda`), and the
reason they had stayed on the CPU stopped being true: "they run once per render and cost 0.0-0.9 s"
described a single 77-token window, but a real prompt is two or three windows x positive and
negative x two SDXL towers, which measured **1.7 s** at 1024x1536 - more than a whole 512^2
sampling step. See "### Long prompts" above for why the chunking exists.

- **The batch axis is the chunk**, since one 77-row window is only 77*heads threads. The
  empty-prompt reference `z_empty` rides along as one more batch item, which is also exactly how
  ComfyUI computes it - and it is cached per tower, so a second render pays nothing.
- Three kernels were needed, all CLIP-specific and all listed in BACKEND.md 2A:
  `attn_causal_batched` (vulkan only - `Backend.attn` already had a `causal` flag), plus ungated
  `gelu_quick` and `gelu_erf` on both. ⚠️ **CLIP's tower is the only CAUSAL encoder here** - every
  other one (SigLIP, Snowflake, the ViTs) is bidirectional, and a non-causal CLIP encodes every
  prompt and renders every image, so there is nothing to observe if it is wrong. The Vulkan test
  asserts the non-causal kernel *disagrees*, precisely so a swapped kernel cannot pass.
- ⚠️ **The two towers of ONE checkpoint use different activations** (CLIP-L quick-GELU, CLIP-G
  erf-GELU) and the three GELU forms here agree to ~1e-2 - close enough to look right, far enough
  to shift style. Compared value by value, not in aggregate, for that reason.
- ⚠️ **Precision is not symmetric between the arms.** Vulkan follows `qwen3_gpu`'s convention
  (f32 by default, tensor-core f16 only under `--encoder-f16`) and holds **1.1e-5** rel L2 against
  the CPU forward; CUDA has only a tensor-core entry point at these widths, so it runs f16
  regardless and lands ~1e-3 - the regime `sd_unet_cuda` already runs the denoiser in. So a `cuda`
  render and a `cpu` render of one seed differ slightly at the conditioning.
- ⚠️ **`Session.runClip`'s CPU fallback is an ALLOW-LIST, and making it a catch-all hid a real bug
  behind an error naming the wrong stage.** `CUDA_ERROR_ILLEGAL_ADDRESS` is **sticky**: once a
  kernel faults, every later call in that context fails too. The first version caught *any* error,
  logged a warning, re-ran the encode on the CPU (which succeeds — it needs no device) and returned
  normally; the render then died at the denoiser's first conditioning upload with
  `cuMemcpyHtoD failed: CUDA_ERROR_ILLEGAL_ADDRESS`, three stages past the fault, with the real
  error demoted to a warning. It was reported as "fails after the TE step", which is precisely
  where it did not fail. Fallback is now limited to `UnsupportedDType` (a block-quantized tower
  genuinely has no GPU GEMM) and `OutOfMemory` (context intact, and the CPU needs no VRAM);
  everything else, `CudaError` above all, propagates. **Generalizable: never recover from a device
  fault by running the next stage — the context is already poisoned, and the error that names the
  faulting stage is worth more than a render that limps one stage further.**

⚠️ **GroupNorm's statistics are Welford, not sum-and-sum-of-squares, and that is load-bearing.**
`ops.norm.groupNorm` warns that the shifted `E[x^2] - E[x]^2` form loses catastrophically once
the mean is large relative to the spread — a late VAE decoder block. Measured on the device at
mean 400 / sd 1: Welford tracks the f64 CPU reference to **6.7e-5 rel L2**, which is f32's
representation of `x` itself cancelling down to a quantity of size 1, while the shifted form
has to resolve 160001 - 160000 in f32 and lands ~1% out on the variance. Chan's pairwise merge
over the per-chunk statistics keeps that accuracy at ONE read of the activation, where the
CPU's literal two-pass form would cost two.

⚠️ **Attention head width is the one thing the three GPU arms do not share** (BACKEND.md §2A
has the table). SD's heads are 40/64/80/160 wide; both GPU limits trace to the P@V GEMM tiling
its head dimension in 128-wide blocks (`coopmat.buildGemmAttnOut`, `launchHgemmB`'s
`grid.x = n/128`), so `cuda` uses cuDNN at the true width while `zig-cuda` and `vulkan`
zero-pad up to 128. The padding is exact — a zero dimension contributes nothing to a dot
product and V's zero columns give output columns `head_unpad` drops — so it only costs
arithmetic. The fix is to parameterize those two builders by head width, not to change the
model code. A too-narrow head is not a silent wrong answer on either: Vulkan's grid rounds up
and the padding covers it, CUDA's `grid.x = 0` is `CUDA_ERROR_INVALID_VALUE`.

⚠️ **f16's 65504 ceiling is a real limit on real checkpoints, and it made every SDXL
render at 512² or larger a solid white image** (fixed 2026-08-03). Probed per stage: the
SDXL VAE decoder's *residual stream* reaches **4.2e5** — the last upsample conv's f32
output is 1.26e5 and the next block's 1×1 shortcut reads it — so casting the activation
to f16 gave `inf`, the following GroupNorm turned that into NaN **through its mean**, and
100% of the output was non-finite, which `planarF32ToRgb8` clamps to white. No error
anywhere. SD1.5's VAE peaks near 7e3, which is why the family this code was written
against never showed it, and it is the same reason ComfyUI decodes the SDXL VAE in fp32
(or ships the "fp16-fix" weights) by default.

- The fix divides *only the residual-reading convolutions* (the 1×1 shortcut and the
  level upsamples) by **256** before the cast and multiplies the f32 result back:
  `opConvF16Scaled` / `opMatmulCoopF16WScaled`, `convIntoScaled`, and
  `residual_act_div` in both `sd_vae_cuda` and `sd_vae_gpu`. A power of two makes it
  **exact** (an exponent shift; f16 keeps all 11 mantissa bits) and it is free — both
  halves fold into kernels that already read every element. ⚠️ Those two kernels'
  spare `f32` param is now load-bearing: **every caller must pass 1.0, not 0.0.**
- **Every other convolution here reads a GroupNorm output (peak 67 measured), so it is
  left alone** — scaling it would only cost precision at the bottom of the range, since
  the divisor's real cost is that true values below `256·6e-8 = 1.5e-5` underflow to
  zero (1e-10 of a residual whose peak is 1e5).
- ⚠️ **Three false leads, recorded so nobody re-walks them.** The failure looked
  *intermittent* — the same size passing then failing — which read as a race or a
  stale-device-memory bug; it was a fresh random `z` per iteration landing either side
  of the threshold. It looked *magnitude-independent* (a ×7.7 latent passed where ×1
  failed) for the same reason. And it was **not** the attention path: swapping the
  cuDNN arm for hand-PTX changed nothing, because both cast through the same
  conversion kernel. A per-stage non-finite probe found it in minutes after ~an hour
  of hypotheses — for a "100% of the output is NaN" symptom, bisect the stages first.
- **`sd-cuda-test`'s VAE check is now a size × magnitude sweep** for this reason
  (12×10 … 128×128, z at ×1 and ×7.7), and its reporter names *which side* went
  non-finite and each side's max |v| — a bare `rel L2 nan` does not distinguish "the
  reference overflowed" from "the kernel did". It also detects the checkpoint's family,
  so an SDXL file exercises the sweep. ⚠️ Its reference decode must NOT run on an arena
  (the CPU decoder frees every intermediate, an arena defers all of it to scope exit,
  and the sweep got OOM-killed at 128²).

⚠️ **`ensureDeviceBuffer` inside a recording batch is a live hazard, and it cost a wrong
decode.** The SD VAE's activation sizing has to mirror the loader's own width bookkeeping: the
first resnet of a level reads the PREVIOUS level's (wider) input at the NEW (doubled)
resolution — 512 channels over 1920 positions where the level's own output is 256 — so sizing
off `block_out_channels` alone under-allocates by 2x. That surfaced not as an allocation
failure but as a **rel L2 of 0.24** against the CPU decoder, because growing the buffer
discarded the activation it held and freed memory that recorded-but-unsubmitted dispatches
still referenced. Both SD files now allocate everything BEFORE `beginBatch` and assert
capacity rather than grow.

⚠️ **`Context.smallBuffer` caches by POINTER and never re-uploads**, so a per-forward vector
written back into the same host array would silently keep the first forward's values. The
Vulkan UNet folds each ResBlock's timestep-embedding projection into its convolution bias —
exact, since a conv bias is already a per-channel constant over positions, and it saves a
read-modify-write of the full activation per ResBlock — and that folded vector changes every
forward, hence `opMatmulCoopF16WDev` and one packed device buffer. The CUDA arm adds the
projection with `opAddBiasRows` instead, because its GEMM entry points take a host bias slice
and reworking that plumbing is not worth ~2% of a ResBlock.

⚠️ **`ctx.independent(n)` counts DISPATCHES, not calls, and a coop GEMM is three of them
sharing `x_h16`/`y_pad`.** Marking three q/k/v GEMMs independent therefore both let them
clobber each other's scratch AND dropped the barriers *inside* the first one. That was the
first bug in the Vulkan UNet, and it rendered a blue streak pattern rather than erroring.

⚠️ **`head_unpad` writes binding `b`; `head_pad_h16` writes `d`.** A kernel bound to the wrong
slot writes nothing, which reads as "attention returned zeros" — rel L2 of exactly 1.0. Worth
knowing as a signature, because it looks like a numerics failure and is not.

**Every new SPIR-V kernel is pinned against the CPU op it reproduces** (`sd_unet_gpu`'s test
block, needing `-Dintegration` AND the `testdata/gpu-tests` marker): GroupNorm in both the
mean-0 and mean-400 regimes, the erf-GELU gate value by value, cross-attention at unequal q/kv
lengths, self-attention at all four SD head widths, the convolution at stride 1 / stride 2 /
fused 2x upsample, the channel concat, the head pad/unpad round trip, and a whole-decoder
parity test for the VAE. That harness localized every bug above in seconds; the one bug found
by looking at a render instead cost far more.

**The CUDA kernels are checked by `TensorPencil sd-cuda-test [<ckpt>] [libs]`**, not by a unit
test: the test binary brings up no CUDA context, so validation here is a CLI command, matching
`cuda-vae-test` / `cuda-dit-test`. It checks each kernel against its CPU op on random data,
then both whole stages against their CPU forwards, and **exits non-zero if any check fails** so
it works as a gate. Run it in BOTH arms — `libs` takes the cuDNN attention path, and the output
makes the difference visible: it reports "runs at 40/64/80/160" where the default hand-PTX arm
reports "runs at 128/128/128/256".

⚠️ **One tolerance legitimately differs between the two backends**: `geglu` holds 1e-6 on
Vulkan and needs 2e-5 on CUDA, because every kernel in `cuda/elt.zig` computes exp as
`ex2.approx(x*log2e)` — an approximate instruction, per that file's header — where SPIR-V's
`@exp` is not. Measured 6.6e-6, two orders below the f16 GEMM error either side of it. Do not
"fix" it by tightening the bound.

## Z-Image ("zit"): a fourth family

⚠️ **It is not Cosmos, and it is not its own model in ComfyUI.** Z-Image is Tongyi's
6B `NextDiT`, which ComfyUI runs through `comfy/ldm/lumina/model.py` — the
Lumina-Image-2.0 model — switched into a different shape by `z_image_modulation=True`
/ `pad_tokens_multiple=32` / `time_scale=1000` / `rope_theta=256`, and selected purely
by `dim == 3840` (`supported_models.py::ZImage`, a subclass of `Lumina2`). Anything in
`models/zimage.zig` that reads as "Lumina but different" is that flag.

Landed 2026-08-06, CPU only so far. It cost far less than the SD family did, because
three quarters of it already existed: flow matching, 3-axis interleaved RoPE, RMSNorm,
SwiGLU and — the big one — **a Qwen3-4B text encoder**, which krea2 already runs.

| piece | what it is | reuse |
|---|---|---|
| tokenizer | Qwen2 BPE, 151936 | `core/tokenizer.zig` **unchanged** |
| text encoder | Qwen3-4B, hidden 2560 | `qwen3.TextEncoder`, new `.zimage` `Variant` |
| RoPE | 3-axis `(32,48,48)`, **theta 256** | `ops.rope.fluxFreqs`, krea2's is theta 1000 |
| VAE | `AutoencoderKL`, **16** latent channels | `sd_vae.Decoder`, new `flux` config |
| sigma table | `ModelSamplingDiscreteFlow`, shift **3.0** | new `SigmaTable.discrete_flow` |
| denoiser | 30 blocks + 2+2 refiners | new: `models/zimage.zig` |

**Six things in the block are NOT krea2's**, and each is a silent wrong answer rather
than an error — they are enumerated in `zimage.zig`'s module header and worth reading
before touching it: sandwich norms (a second RMSNorm on the sublayer's *output*, inside
the residual), `tanh`'d gates, modulation with **no shift** (`x * (1 + scale)`, four
chunks not six), **no SiLU before the block AdaLN linear** (the final layer keeps its
SiLU — which is why the checkpoint numbers them `adaLN_modulation.0` and
`.1`), a weightless **LayerNorm** for the final norm, and a **negated output**.

⚠️ **The patch feature order is `(ph, pw, c)` — channel FASTEST — where krea2's is
`(c, ph, pw)`.** Getting it backwards is a pure permutation: every norm, every
magnitude and every per-stage statistic still matches, and only the image is wrong.
The same class of bug as the SD planar/channel-last mixup, which this file already
records once.

⚠️ **Two learned pad tokens, and the two halves pad DIFFERENTLY.** Both the caption
and the image are padded up to a multiple of 32 with `cap_pad_token` / `x_pad_token` —
but the caption's pad tokens are appended *before* its position ids are built, so they
continue the `1..n` ramp, while the image's positions are `F.pad`ded with **zeros**, so
every image pad token sits at `(0,0,0)`. Making them consistent either way changes the
attention pattern of every padded render.

⚠️ **The Q/K norms use a DIFFERENT epsilon from the block norms** — `RMSNorm(head_dim,
elementwise_affine=True)` is built with no `eps`, so torch falls back to
`finfo(float32).eps` (1.19e-7) where the block norms use 1e-5. ComfyUI's own fused path
spells this out; nothing in the checkpoint does.

### What was generalized rather than copied

- **`qwen3.TextEncoder` gained a `Variant`.** Both variants run the *same* 35 layers of
  the same 2560-wide body and differ in exactly three things: the tensor prefix (krea2
  ships the Qwen3-**VL** checkpoint, whose language model is nested under
  `model.language_model.`), the RoPE theta (5e6 vs **1e6**), and the tap list. ⚠️ The
  theta is the dangerous one — not recoverable from the weights, and the wrong value
  encodes perfectly finite nonsense. Z-Image's conditioning is `hidden_states[-2]` with
  the final norm skipped, which in this file's convention is tap 35 — i.e. **exactly
  krea2's last tap**. The two GPU encoders now read the tap list and theta off the
  encoder instance rather than module constants, so neither can silently apply krea2's.
- **`sd_vae` gained a `Naming` enum, and it is DETECTED, not configured.** ⚠️ The very
  same 16-channel Z-Image VAE ships both ways: the official `ae.safetensors` the
  ComfyUI template points at is **LDM**-named, while `z-image-turbo.vae.safetensors` is
  **diffusers**-named. A config field would be right for one and wrong for the other.
  The probe is on the *middle block* — `decoder.conv_in` is spelled identically in both.
  The two schemes also index up-blocks in **opposite directions**, and the Flux-lineage
  VAE has **no `post_quant_conv`** at all (BFL dropped it), which under the old loader
  was a hard `MissingTensor` on every Flux/Z-Image VAE in existence.
- **`SigmaTable.discrete_flow`**, not `.flux` with a converted shift. The two formulas
  are algebraically the same function, but the table is **1000** rungs against 10000
  and torch evaluates this one as a single tensor division where the flux form is
  `scalar / tensor` (which lowers to `reciprocal * scalar` and rounds twice). Both
  differences are invisible to Euler and decisive for an SDE sampler, whose Brownian
  tree keys on the sigma quantised to 1e-6.
- **`Options.explicit_shift` + `defaultShift(fam)`.** Z-Image trained on shift **3.0**
  and `Options.shift` defaults to krea2's 1.15; at 8 steps the wrong one is a valid
  schedule with the steps in the wrong places, not an error. Same for
  `defaultComponentPath`: handing krea2's Qwen3-VL encoder to Z-Image resolves to
  nothing, which is the failure a joined SD1.5 checkpoint already hit once.
- **`ops.matmul.materializeF32`** and **`ops.norm.layerNormUnit`** were hoisted out of
  `dit.zig` / added, so both denoisers share them.

### Two bugs the integration surfaced

- ⚠️ **`sdTile` hardcoded `sd_vae.latent_channels` (4).** The same decoder body serves
  SD's 4-channel latent and Z-Image's 16-channel one, so it transposed the first
  quarter of the latent and read past it for the rest — rendering as a band of colour
  noise across the top of an otherwise flat grey image, with the assert compiled out in
  ReleaseFast. The count is derived from `sub.len / (th*tw)` now, as `vae_tiled.decode`
  already did, and the test runs at **both** widths (a 4-channel-only test could not see
  it — 4 was the hardcoded value).
- **`std.json` parks an integer that does not fit `i64` in `number_string`.** The
  Z-Image sigma table's FNV-1a hash is exactly such a value, so reading `.integer`
  unconditionally panicked — while passing for the other two tables, which is how it
  stayed latent until a third one existed.

### Measured

Reference is **ComfyUI**, via `tools/gen_zimage_fixtures.py`. ⚠️ The DiT reference is
built at the detected config but **truncated to 2 of 30 layers**: a fp32 copy of the
6.1B denoiser is 24.6 GB and does not fit here, and a bf16 reference would put a ~1e-2
floor in the *reference* — coarser than several of the bugs the fixture exists to catch
(a wrong pad-token position and a wrong rope axis both survive 1e-2). Truncating buys an
**exact fp32 reference on real weights at real width**; what it does not cover is the
loop bound, which the end-to-end render does. The generator asserts ComfyUI's own
`detect_unet_config` output against the constants it writes, so the config cannot drift.

| stage | agreement |
|---|---|
| velocity, whole forward, real weights (both latent sizes) | **1.6e-6** rel L2 |
| `t_embedder` / `context_refiner` / `noise_refiner` | 1.5e-4 / 2.3e-4 / 2.0e-4 — the fixture's own f16 storage floor |
| text encoder (Qwen3-4B penultimate) | **1.0e-4** — also the f16 floor |
| tokenizer + chat template | exact ids |
| VAE decode (16-ch, both namings) | **2e-6** rel L2 |
| sigma table vs `model_sampling.sigmas` | **bit-exact**, all 1000 entries (FNV-1a over raw f32 bits) |
| 9 schedulers x 4 step counts vs `calculate_sigmas` | < 4e-7 rel |

### End to end, against a real ComfyUI render

⚠️ **The DiT reference is truncated to 8 of 30 layers**, and 8 is what fits rather
than what was wanted: a fp32 copy of the whole 6.1B denoiser is 24.6 GB, and 16 layers
(~14.8 GB before torch's overhead) was OOM-killed inside a 22 GB cgroup bound. What
that buys is an **exact fp32 reference on real weights at real width**. The depth trend
is the useful part: **1.6e-6 at 2 layers, 3.5e-6 at 8** — the disagreement grows
roughly *linearly* with depth rather than compounding, which is what says the block is
right rather than merely close.

Then a like-for-like render, at the user's own ComfyUI settings pulled out of the PNG's
embedded workflow (`unstableRevolution_V2Fp16`, 512x768, 9 steps, cfg 1, euler +
**beta**, seed 80085, `z-image-turbo.vae`, a 258-token prompt):

| | PSNR | SSIM |
|---|---|---|
| **TensorPencil (f32) vs ComfyUI bf16** | **30.81 dB** | **0.9632** |
| ComfyUI bf16 vs ComfyUI fp16 — *its own precision floor* | 24.49 dB | 0.8900 |
| TensorPencil (f32) vs ComfyUI fp16 | 24.64 dB | 0.8928 |

⚠️ **Read the middle row first.** Z-Image Turbo at 8-9 steps is extraordinarily
precision-sensitive: ComfyUI disagrees with *itself* by 24.5 dB purely from the
denoiser's dtype. TensorPencil lands **6.3 dB closer to ComfyUI's bf16 render than
ComfyUI's own fp16 render does**, i.e. comfortably inside the reference's own envelope.
Without that control the 30.8 dB figure is uninterpretable — this is the isolation
measurement, not a footnote.

⚠️ **A 14 dB "disagreement" that was entirely the harness.** The first comparison read
14.39 dB with visibly more contrast and saturation on the reference side. Cause:
`tools/render_zimage_ref.py` drives `first_stage_model.decode` directly (it has to —
`VAE.decode` picks its own dtype and hands bf16 samples to fp32 weights), and that
**skips `VAE.process_output`, which is `(x + 1) / 2`**. Clamping the raw [-1, 1]
decoder output to [0, 1] crushes every negative to black, which looks like a plausible
image with punchier colour rather than like an error. Same family as the
`VAEDecodeTiled` trap already recorded above: **a reference is a piece of code and can
be the thing that is wrong.** The structure matching pixel-for-pixel while only the
tone differed was the tell — that signature means the trajectory agreed and only the
output mapping did not.

**Defaults**, from ComfyUI's own `image_z_image_turbo` template: 1024², **8 steps**,
**cfg 1.0** (no negative branch), `euler` + **`simple`**, shift 3.0, and the three files
`unstableRevolution_V2Fp16` / `qwen_3_4b.safetensors` / `ae.safetensors`. The template's
sampler is actually `res_multistep`, which this engine does not have; `euler` is the
closest available and is what the defaults select.

### On the GPU

**Every backend runs the whole pipeline** — text encoder, the 30-block trunk
(`zimage_gpu.zig` / `zimage_cuda.zig`) and the VAE decode. Measured at 1056x1584 /
9 steps / cfg 1 / euler+beta on a 3090: **`cuda` 41.3 s (2.94 s/step)** and **`vulkan`
50.5 s (4.55 s/step)**, landing **29.42 dB** and **28.96 dB** (SSIM 0.96) against a real
ComfyUI render of the same workflow. The two arms agree with each other at **34.20 dB /
SSIM 0.986**, and both sit inside ComfyUI's own **24.49 dB** bf16-vs-fp16 envelope — read
that control first, because this model at 9 turbo steps is precision-dominated. Unit-level:
`attn_full` matches `ops.attention` at 4.9e-7 and the Vulkan forward matches
`zimage.DiT.predict` at 2.15e-4 (`attn_full`) / 2.19e-4 (tensor-core). No new kernel was
needed on either backend; BACKEND.md §2B has the mapping.

⚠️ **The two backends take DIFFERENT modulation tables, and that is forced, not sloppy.**
CUDA has no standalone `modulate` — only `rmsMod` — so the pre-norm weight must arrive
folded into the scale (`modulationTableFolded`); Vulkan has a separate `modulate` and
takes the unfused one. Both are derived from one AdaLN evaluation so they cannot drift.

**Five things it cost, and four are the same lesson — a claim that was never checked:**

- ⚠️ **`qwen3_gpu.encode` handled fp8 only**, and Z-Image's `qwen_3_4b.safetensors` is
  **bf16**: it fell through to `opMatmul` with `dtype_f8 = false`, i.e. the f32 pipeline
  reading bf16 bytes. Every GEMM garbage, conditioning non-finite, render solid white,
  no error anywhere. ⚠️ The trap: that file's `wcode` helper *does* read bf16 natively —
  but that is the **LM's GEMV** path, not the encoder's GEMM. **Read the call site, not
  a neighbouring comment.** `qwen3_gpu.supportsWeights` gates it now.
- ⚠️ **`sd_vae_gpu`'s mid-block scores plane had to become f32.** The Flux/Z-Image VAE's
  attention **logits reach 9.95e6** on a 12x10 latent where SD1.5's reach **8.3** — 152x
  past f16's 65504, and even without overflow f16's quantum up there is ~8000, which
  destroys a softmax whose differences are O(1). Result: solid white, no error.
  ⚠️ Everything upstream is finite and *smaller* than SD's (489 vs 929 at the last
  level), so only the logits themselves show it — and the 4-channel-only test could not,
  because 4 was also the hardcoded channel count. Both are fixed: the plane is f32, the
  test runs at 4 **and** 16, and `estimatePeakBytes` reports 4 bytes so the ladder tiles
  when a whole-image plane will not fit. SD1.5 still matches the CPU decoder at 1.1e-3.
- ⚠️ **`attn_out` runs its OWN online softmax.** Inserting a `softmax_rows` before it
  exponentiates twice — finite, plausible, wrong by rel L2 0.26. `dit_gpu`'s f32 branch
  has no softmax call for exactly this reason, and that absence reads as a missing step
  until you read the kernel.
- ⚠️ **`attn_full` trips the GPU watchdog above ~768px** (`error_device_lost`, not a
  slow render), so the tensor-core scores/PV path is a *requirement*, not an
  optimization. Landing correctness-first was still right — it is what the fast path was
  then validated against — but the label was wrong.

- ⚠️ **`sd_vae_cuda` had the SAME f16-scores bug and I did not fix it with the Vulkan
  one** — so the first CUDA render was white too, for a reason I had already diagnosed
  and written down an hour earlier. **When a defect is in a shared component with a
  per-backend twin, fix or check the twin in the same pass.** Its resolution is
  different (`be.attn`'s f32 online softmax materializes no plane at all, which suits
  one attention at latent resolution) but the diagnosis was identical.

### The VAE decode: 12.0 s -> 1.5 s, and 7.7 GB -> 5.1 GB (2026-08-06)

Measured at 1056x1584 (a 132x198 latent, so the mid-block attention runs at
**seq = 26,136**). Both halves were self-inflicted, and both by the same mistake:

⚠️ **A measurement taken in one regime, generalized to all of them.** This file
used to say `be.attn`'s one-thread-per-query weakness "does not bite here: this is
ONE attention per decode at *latent* resolution, and the decode ladder tiles above
64² anyway". The ladder does **not** tile when the whole image fits, which at
1056x1584 it does — so 26,136 queries each streamed the entire 53 MB K matrix with
no reuse. That was **12 s of a 21 s render.**

- **`opAttnTC` was rejected on the wrong arm.** The rejection was real — the
  Flux/Z-Image VAE's logits reach **9.95e6**, 152x past f16 — but it was measured
  at a **12x10** latent, where `opAttnTC` takes the batched path that stores an f16
  scores plane. At production seq it routes elsewhere: cuDNN's fused SDPA under
  `.libs` (Q/K/V f16, softmax internal and f32 — the logits never land in f16), and
  `opAttnTCFlash` under hand-PTX (which **does** store an f16 band, so the rejection
  stands there). `sd_vae_cuda.attn` now picks per arm.
- **The estimate charged 2.73 GB for a plane CUDA never allocates.**
  `estimatePeakBytes` takes `scores_resident` now, because the backends genuinely
  differ: `sd_vae_gpu` materializes an f32 plane, both CUDA arms stream. Charging
  it made the ladder **evict 2.4 GB of resident DiT weights** before a decode that
  needed no room, and re-stream them next image.

⚠️ **A third of the activation VRAM was one buffer sized off the wrong width.**
`x`, `t` and `u` were all allocated at the ladder's widest tensor. `x` (the running
activation) and `t` (a norm's output) do reach it — 256 channels at FULL image
resolution, 428M floats — but `u` only ever receives a convolution's **output**, so
it peaks one level narrower (128 channels, 214M). `sd_vae.activationElems` returns
both widths now and both GPU decoders use them: **-856 MB**, bit-identical.

**Then f16 activation STORAGE** (`sd_vae.Config.act_f16`) halved what was left. The
arithmetic is unchanged — every conv GEMM already ran f16 operands with an f32
accumulator, and the norms still reduce in f32 — only the two 1.71 GB buffers narrow.
Five storage-format twin kernels (`gn_stats_h16`, `gn_apply_h16`, `add_h16`,
`bias_compact_h16`, `im2col_sd_h16`) plus `src_f16`/`dst_f16` on `convIntoPrec`.

| backend | decode | est peak | note |
|---|---|---|---|
| **`cuda`** | 12.0 -> **1.4 s** | 7.7 -> **2.2 GB** | cuDNN SDPA + f16 storage |
| `vulkan` | 2.1 s | 6.9 GB | f32 storage AND an f32 `seq²` plane (2.73 GB of it) |
| `zig-cuda` | 13.2 s | 2.2 GB | f16 storage, but still `be.attn` |

**3.4x less VRAM and 8.6x faster, for 1/255.** The f16 render is **68.5 dB / SSIM
0.9999** against the f32 one (max difference one 8-bit level) and unchanged at 30.00
dB against ComfyUI. Component-level it moves the decode's rel L2 from 4.5e-4 to
8.4e-4 against the CPU decoder — 6x inside the harness tolerance.

⚠️ **RANGE gates this, not precision, and it is per architecture.** These buffers
carry the residual stream: measured peak ~489 for the Flux/Z-Image VAE and ~7e3 for
SD1.5, both far inside f16's 65504 — but **4.2e5 for SDXL's**, which is the whole
reason `residual_act_div` exists. An f16 *buffer* cannot hold a value that divisor
was invented to sneak through a *cast*, so the two are alternatives, not layers:
`act_f16` is on for `flux`/`sd15` and off for `sdxl`, and `convIntoPrec` asserts they
are never combined.

⚠️ **`estimatePeakBytes` takes the CALLER's storage format, not `cfg.act_f16`.** Only
the CUDA decoder implements f16 storage; reporting the narrow figure to Vulkan would
send the ladder into a whole-image decode that cannot fit — an OOM found the
expensive way, which is precisely what the estimate exists to prevent.

⚠️ **f16 storage FORCES the cooperative conv path, and not forcing it broke SD1.5
completely.** `coop_min_co` (96) is a *performance* threshold — below it the f32 GEMM
wins — but the f32 arm has no f16 storage form. SD1.5's `post_quant_conv` is 4->4,
fell to that arm, and wrote f32 into an f16 buffer: **every pixel non-finite**. Caught
by `sd-cuda-test`'s size x magnitude sweep, which is the argument for that sweep
existing; Z-Image never hits it (the Flux VAE has no `post_quant_conv` at all).

⚠️ **For scale, and because the intuition here is backwards: ComfyUI's own
whole-image decode of this latent peaks at 18,792 MB** (measured with
`torch.cuda.max_memory_allocated`, bf16). We are ~4.4x smaller, and it is why the
reference render for this workflow had to come from `VAEDecodeTiled`. "The VAE uses
too much" is worth measuring against the reference before believing it.

**Verified bit-identical** (PSNR ∞) to the pre-change render at 1056x1584, and the
`12x10` / `40x32` A/B in `zimage-cuda-test` shows the two paths agree to the same
4.5e-4 against the CPU decoder while `opAttnTC` is already **3.7x** faster at 40x32.

⚠️ **Two gaps remain, both named rather than hidden:**
- **`zig-cuda` is 13.0 s** because `opAttnTCFlash` stores its query band's scores in
  **f16** and this VAE overflows that. The fix is an f32 scores band — a new hgemm
  C-store variant plus `softmax_md` / `attn_out` reading f32. Not attempted.
- **Vulkan allocates the whole `seq²` f32 plane (2.73 GB).** `sd_vae_gpu.attn` has
  no query loop, though two orphaned doc comments above `Bufs` ("Scores-plane budget
  before the query dimension is tiled", "Interleaved chunks per row in the two-pass
  softmax") describe constants that no longer exist — so it was tiled once and the
  loop was lost. `attn_scores`/`attn_out` would need a query-offset push constant.

`zimage_cuda` **is** validated now: `TensorPencil zimage-cuda-test [<ckpt>] [libs]`
checks its forward against `zimage.DiT.predict` on real weights, on both attention paths
(2.23e-4 / 2.14e-4 under `libs`, 2.37e-4 / 2.14e-4 under `hand_ptx`), and exits non-zero
on failure.

### A GGUF text encoder

`--text-encoder foo.gguf` works on **every backend** as of 2026-08-06, and the
motivating file is `Qwen3-4B-Q4_K_M.gguf`: **2.50 GB against the bf16
safetensors' 8.04 GB**, and **2065 MB of VRAM against 6738 MB** live during
sampling — which on a 24 GB card is the difference between the encoder being
free and it competing with an 11 GB DiT.

Five things had to change, and four of them were places that had quietly
assumed "encoder == safetensors == one dtype":

- **`enc_st` was a `?SafeTensors`.** Side-file components now go through
  `Container` (renamed from `DitContainer`, since it serves every component),
  which opens by **magic**. A `.gguf` encoder previously died with
  `InvalidHeader` from the safetensors parser — the exact useless error
  `Container.open` already existed to prevent for the denoiser.
- ⚠️ **`ComponentSpec.probe` became `probes`, a LIST**, because a component's
  own tensor names differ by container. `canonicalName` rewrites llama.cpp's
  `blk.N.attn_q.weight` to `layers.N.self_attn.q_proj.weight`, but there is no
  `model.` prefix to restore — so the GGUF spelling is `embed_tokens.weight`
  where safetensors is `model.embed_tokens.weight`, and with a single probe the
  resolver found nothing and reported `ComponentNotInCheckpoint`.
- ⚠️ **The config now comes from the CHECKPOINT when the checkpoint states it.**
  `loadVariant` calls `Config.detect` for a GGUF, which reads dims, the bare
  prefix and — the one that matters — `rope.freq_base`. The `Variant` keeps only
  what a checkpoint cannot know: **the tap list**, since which hidden states to
  keep is a property of how the *diffusion* model was trained, not of the
  encoder. That split is the whole rule. (It also makes theta *recoverable* for
  a GGUF, where this file's Z-Image section warns it is not for safetensors.)
- ⚠️ **`TextEncoder.embed_bytes` became `embed: Weight`, and this was three
  bugs, not one.** Qwen3-4B-Q4_K_M's `token_embd` is **q6_k**; the loader
  hard-rejected anything but bf16, and *all three* gather sites (CPU, CUDA,
  Vulkan) open-coded `bytes[id * hidden * 2 ..]` with `convertToF32(.bf16, ...)`.
  All three now call the dtype-generic `embedTokens` that already existed.
- **GPU GEMM arms.** `qwen3_cuda.wgemm` routes block quants to `opMatmulQuant`
  and `qwen3_gpu.gemm` to `opMatmulCoopQuant` — the LLM's own prefill paths,
  which suit an encoder exactly (one call over the whole prompt, so the
  per-call weight dequant amortizes).
  - ⚠️ **Deliberately NOT the MMQ path**, which `mmqPipeFaster` would select for
    every q4_k matrix here and which the LLM prefill *does* take. MMQ quantizes
    the **activation** to q8_1 (~0.5% relative). This is a conditioning tensor:
    computed once per render, then steering every sampling step. The LLM makes
    that trade because there the same GEMM runs per token, forever.
  - ⚠️ **Vulkan's `wcode` maps every non-f8/bf16 dtype to `.f32`**, so a block
    quant reaching `gemm`'s fallback would be read as f32 — the identical
    silent-garbage shape as the bf16 bug recorded in that file's
    `supportsWeights` comment. The block-quant arm is checked *before* `wcode`.

⚠️ **Two GGUF dialects reach this loader, and the second states almost
nothing.** llama.cpp files (Unsloth's `Qwen3-4B-Q4_K_M.gguf`) carry the full
`qwen3.*` hyperparameter block and `blk.N.` tensor names; **ComfyUI-style files**
(city96's converter, e.g. `qwen_3_4b-q8_0.gguf`) carry **3 KV entries** —
`general.architecture`, `quantization_version`, `file_type` — and already-HF
tensor names, with the norms themselves quantized (q8_0 vectors, bf16 per-head
q/k norms). `detectGguf` falls back to matching a preset by embedding shape.
- ⚠️ **Where the file is silent the Variant wins, and `rope_theta` is why.** The
  fallback GUESSES plain-Qwen3's 1e6, which is right for Z-Image and silently
  wrong for krea2's Qwen3-VL (5e6) — unrecoverable from weights, and finite
  nonsense when wrong. `statesRopeTheta` asks whether the file actually declared
  one; if not, `loadVariant` keeps the Variant's. "The checkpoint wins for what
  it states" is not "the checkpoint wins".

**Validated by `TensorPencil te-test [<encoder>] [--ref <encoder>] [--krea2]`**,
which deliberately separates two questions that render comparisons conflate:
*does this backend compute what the CPU computes* (a kernel question) and *does
this quantization change the conditioning* (a format question).

| encoder | cpu | cuda | zig-cuda | vulkan | vs bf16 |
|---|---|---|---|---|---|
| `Qwen3-4B-Q4_K_M.gguf` (q4_k) | 1354 ms | **144 ms** / 1.3e-4 | 221 ms / 1.3e-4 | 947 ms / 1.3e-4 | 3.4e-2 |
| `qwen_3_4b-q8_0.gguf` (q8_0) | 919 ms | 151 ms / 1.4e-4 | 210 ms / 1.5e-4 | 1010 ms / 1.5e-4 | **6.4e-3** |
| `qwen_3_4b.safetensors` (bf16) | 793 ms | 136 ms / 3.7e-3 | 187 ms / 3.7e-3 | 127 ms / 3.7e-3 | — |
| krea2's Qwen3-VL (fp8) | 1628 ms | 161 ms / 1.3e-4 | 207 ms / 1.4e-4 | 384 ms / 1.4e-6 | — |

⚠️ **The two quantizations are NOT the same trade, and only a render says so.**
The conditioning deltas differ by 5.4x; the images differ by 8 dB:

| encoder | file | live VRAM | vs ComfyUI | vs the bf16-encoder render |
|---|---|---|---|---|
| bf16 | 8.04 GB | 6738 MB | 29.4 dB | — |
| **q8_0** | 4.27 GB | **3634 MB** | **30.0 dB** | **30.8 dB** |
| q4_k | 2.50 GB | 2065 MB | 21.8 dB | 22.2 dB |

**q8_0 is free** — it lands *inside* this model's own 24.49 dB bf16-vs-fp16
floor and scores marginally better against ComfyUI than the bf16 encoder does,
i.e. indistinguishable, at half the size. **q4_k is not**: 22 dB is below that
floor, so it gives a **different, equally valid** image of the same prompt (same
composition, same palette, different detail) rather than a cheaper route to the
same one. The three backends agree with each other at 35.7-37.5 dB, so what
moves is the format, not any one arm. Recommend q8_0; offer q4_k as a
VRAM-constrained choice with that caveat stated.

### Getting to ComfyUI's speed: what was actually slow

**2.00 s/step against ComfyUI's 1.836 — 91.8%, up from 58%** (3090, 1056x1584 / 9 steps),
renders unchanged in kind at 29.38 dB / SSIM 0.9592. BACKEND.md §2B has the full ladder
and every isolation; what belongs here is the *shape* of the mistake, because it is one
this file has recorded in other forms and will again.

⚠️ **The whole-render profile pointed at the wrong component, and it was not lying — it
was answering a different question.** It put **88% of the step in `matmul`**, which reads
as "the GEMMs are slow, so this is kernel efficiency and there is no cheap fix". Two
things were wrong with that reading. `opGemmBf16` is `ptic()`-scoped, so the two
f32<->bf16 streaming passes wrapped around every GEMM were counted **as** GEMM time; and
the norms sat inside a 7% `elt` bucket that looked far too small to be worth opening.

`TensorPencil zimage-cuda-bench` replays one step's device work at the real shapes with no
checkpoint and no sampler, split GEMM / attention / elementwise — and inside `opGemmBf16`,
splits the GEMM from its conversion passes via `backend.bench_gemm_only`. It inverted the
profile:

- **cuBLASLt was already at ~92% of the card's bf16 roof** (65.2 TFLOP/s pure GEMM).
  There was never anything to win there.
- **`qkNorm` was 654 ms of a 2556 ms step at 37-47 GB/s** on a 936 GB/s card — a quarter
  of the render, in two calls nobody had ever timed, because "a norm obviously costs
  nothing next to a GEMM".

⚠️ **The metric that made it legible is achieved BANDWIDTH, not share of time.** These
kernels do almost no arithmetic, so bytes-moved / seconds is a number with a known
ceiling — and every *other* elementwise kernel in the same block sat at 500-750 GB/s,
which is what turns "elementwise is 30% of the step" into "these two kernels are broken".
A share-of-time table cannot say that: it has no notion of what the op *should* have cost.

The defect: `qk_rmsnorm` gave each **thread** a whole row, so a warp's 32 loads were 32
sector fetches `4*dim` apart. `qk_rmsnorm_warp` gives each **warp** a row (butterfly
shuffle, no shared memory) — 17x and 10x on the two call sites. ⚠️ The row sum becomes a
tree where it was serial, so it is **not** bit-identical; it is the more accurate of the
two, and `rms_mod_par` next door already reduced that way. `rows < 512` still takes the
block-per-row kernel, because LLM decode norms 1 x 2560 and 8 rows per block would leave
82 SMs idle.

⚠️ **The same misattribution repeated one level down, in the warm-up.** Step 1 cost ~5 s
more than a steady step, and that was "the weight upload" — i.e. PCIe, i.e. physics.
Splitting `TP_WARMUP_PROFILE`'s single number into `fill` and `slot-wait` said otherwise:
**`fill 4.88s, slot-wait 0.04s`**. PCIe was idle 99% of the time while ONE thread filled
the pinned staging slots at 2.37 GB/s — against 4.3 GB/s cold and 26 GB/s page-cached for
the same file under `dd`. `Context.FillPool` fans that over 4 threads (positional reads
need no ordering between them): step 1 5.1 -> 2.6 s, render **bit-identical**.

**The generalizable lesson, and the real content of this section:** a profiler bucket is a
*label a programmer chose*, not a measurement of a component. Before concluding that the
expensive bucket is the problem, check what the timer actually spans (`ptic`/`ptoc` here
wrapped an *op*, not a kernel) and give each part a ceiling to be judged against. Both
fixes here were in code the profile said was 7% and 0% of the problem.

⚠️ **Found, NOT caused, in the same pass: `cuda-dit-test` on a bf16 krea2 checkpoint
fails** — rel L2 0.141 against the CPU forward, and a 256px forward taking ~11.5 s/step.
A/B'd across the `qkNorm` change (0.14061 with the old kernel, 0.14098 with the new), so
it is pre-existing and unrelated to any of the above. Unfixed, and worth someone's time:
it means krea2's bf16 CUDA arm has no working validation today.

## Samplers and schedulers

Two independent axes, and as of 2026-08-03 both are selectable — before that there was
one sampler (Euler, called inline) and one schedule per family. They live in two files
on purpose, because the concerns are orthogonal (any sampler runs on any schedule, which
is why ComfyUI presents them as two dropdowns):

| file | owns | surface |
|---|---|---|
| `core/schedule.zig` | **where the steps go**: all nine ComfyUI schedulers + the two families' sigma tables | `--scheduler`, `Options.scheduler` |
| `core/sampler.zig` | **how to step**: Euler and DPM++ 2M SDE (midpoint / Heun) | `--sampler`, `--sde-eta`, `--sde-s-noise`, `Options.sampler` |

Both also appear as Settings dropdowns in the GUI, and both are recorded in the saved
PNG's AUTOMATIC1111 `parameters` block. ⚠️ **Both of those metadata fields used to be
hardcoded** — a literal `Sampler: Euler`, and a `Schedule type` derived from the
architecture — which was only ever right because neither was selectable. A reader
(including ComfyUI's own metadata importer) re-renders from that block, so a stale name
there is a wrong answer nothing else would catch.

### The schedulers

`schedule.SigmaTable` is the family-neutral view of `model_sampling.sigmas` that a
scheduler reads: `.flux` (krea2 — 10000 entries from the shift formula, computed on
demand) or `.discrete` (the SD family's 1000-rung beta ladder, borrowed from
`SdModels.sigma_ladder`). `Session.sigmaTable`/`scheduleWith` dispatch; `Scheduler.defaultFor`
keeps the pre-existing behaviour when the caller does not choose — **`simple` for krea2,
`normal` for SD**, which are genuinely different defaults, so `null` (the GUI's "Model
default") is a real choice and not a synonym for `normal`.

All nine are ported: `normal`, `karras`, `exponential`, `sgm_uniform`, `simple`,
`ddim_uniform`, `beta`, `linear_quadratic`, `kl_optimal`.

⚠️ **`ddim_uniform` and `beta` do not return `steps + 1` sigmas.** The first strides the
table (30 requested → 31 delivered), the second de-duplicates quantiles that round to
the same rung (so it can return *fewer*). `generate` therefore takes its step count from
`sigmas.len - 1` and reports a mismatch; looping to `opts.steps` reads past the end of a
shortened schedule. The GUI's per-step total comes from the progress callback, which
carries the real count.

⚠️ **`beta` needs a real numeric routine, not a formula.** `scipy.stats.beta.ppf` is the
inverse regularized incomplete beta; `schedule.zig` has `betaInc` (Numerical Recipes'
Lentz continued fraction) and `betaIncInv` (plain bisection — the CDF is monotone on
[0,1], it converges for roots of any magnitude, and Newton would need guarding against
the infinite density at both ends when `a < 1`, which 0.6 is). Pinned against scipy at
1e-11, with a forward round-trip so a fixture typo cannot make both sides agree on a
wrong answer. `numpy.rint` is round-half-to-**even**, which `@round` is not.

⚠️ **Three torch behaviours had to be reverse-engineered**, none visible in the Python
source, each worth ulps on hundreds of table entries — and per the SDE section below, an
ulp in a schedule value is a *different noise draw*:

1. **`scalar / tensor` in PyTorch is `reciprocal(tensor) * scalar`** — two roundings, not
   one. `flux_time_shift` is exactly that shape; a single f32 division disagrees on
   **3114 of 10000** flux entries, 165 crossing a `round6` cell.
2. **`torch.linspace` for f32 is FMA-contracted.** ATen's kernel is
   `scalar_start + step * i` in `float` with `step` **also** in f32 (not the double its
   `accscalar_t` suggests), which `-ffp-contract=fast` fuses. Hence `@mulAdd`, plus
   torch's halfway split so both endpoints land exactly.
3. **The same formula has two precisions depending on how it is called.**
   `ModelSamplingFlux.sigma(t)` on a *tensor* is f32; the same class's
   `percent_to_sigma` takes a *Python float* and is f64. `sigmaAt` is the second path and
   `offsetFirstSigma` needs it.

**Verified:** both sigma tables are bit-exact to `model_sampling.sigmas` on **every**
entry, checked by an FNV-1a hash over the raw f32 bits rather than by sampling indices —
deliberately, since the reciprocal convention above moves only 1.65% of the flux table
and a seven-index spot check would miss it. Then all **9 schedulers × 2 families × 4 step
counts = 72** combinations against ComfyUI's own `calculate_sigmas` at 4e-7 relative
(max observed 3.0e-7, the floor set by torch's vectorized `logf` inside `interpLadder`),
plus the fraction landing in ComfyUI's own `round6` cell: **68 of 72 place every sigma
in the right cell**, the other four miss exactly one. End to end at 512²/10 steps against
ComfyUI renders: `karras` **45.2 dB**, `beta` **48.4 dB** — both *better* than `normal`'s
36.4 dB baseline, because they are pure formula / pure table lookup where `normal` goes
through the log-space interpolation.

### The samplers

`pipeline.Options.sampler` (`sampler.Kind`) selects `euler`, `dpmpp_2m_sde` or
`dpmpp_2m_sde_heun`. The two SDE variants are one solver differing only in the multistep
correction form; ComfyUI ships both under those names and they give visibly different
images.

**Euler needs no state; DPM++ 2M SDE needs three kinds**, which is the whole structural
difference and where the plumbing lives:

- **Multistep history.** The second-order correction reuses the *previous* step's
  denoised prediction, so `SdeStepper` is stateful — and `pipeline.Snapshot` had to
  grow `sde_old_denoised`/`sde_h_last`, or an unload-while-paused resume would silently
  take one first-order step and stop being bit-identical (the thing `resume_from`
  promises). `SdeStepper.restore` is that path, pinned by a fast test that splits a run
  in half and demands the same bytes.
- **Half-logSNR, per family.** `lambda = log(alpha/sigma)`, and the families compute it
  *differently*: `flow` (krea2/`CONST`) is `log((1-sigma)/sigma)`, `eps` (SD) is
  `-log(sigma)`. `Session.parameterization()` dispatches. ⚠️ Getting it backwards is not
  symmetric: `flow` on an SD ladder takes the log of a negative number and NaNs the
  whole render, while `eps` on a krea2 schedule is perfectly finite and just integrates
  the wrong ODE.
- ⚠️ **`offsetFirstSigma`, without which flow matching is all NaN.** krea2's schedule
  starts at sigma **exactly 1**, where `log((1-sigma)/sigma)` is -inf. ComfyUI's
  `offset_first_sigma_for_snr` nudges it to `percent_to_sigma(1e-4)` = 0.99996833.
  Ordering is load-bearing at both ends: it must run **after** `scaleInitialNoise`
  (ComfyUI scales the starting latent by the *unoffset* first sigma) and **before**
  `Session.denoiser`, which caches a timestep vector per schedule entry.

**The noise is the hard part, and it is a real port, not a `randn`.** ComfyUI's SDE
samplers draw from `torchsde.BrownianTree` via `BrownianTreeNoiseSampler` — one Brownian
path over the *sigma axis*, fixed by the seed alone, queried by interval. That is what
makes the same seed hold its structure across step counts, and it is three pieces, all
now in `tp_core`:

| file | reproduces |
|---|---|
| `core/seed_seq.zig` | `numpy.random.SeedSequence` entropy-pool mixing + `generate_state` |
| `core/brownian.zig` | the dyadic `BrownianInterval(halfway_tree=True, tol=1e-6)`, W-branch only |
| `core/torch_rng.zig` | `torch.randn` per node (already existed) |

⚠️ **Which generator that last row uses is selectable, and A1111 needs the other one.**
`core/noise.zig` dispatches to `torch_rng.zig` or `philox_rng.zig`; the tree's `src` comes
from `SdeStepper.Options.noise_src`, because A1111's pinned k-diffusion builds the tree on
the CUDA tensor's own device where ComfyUI's fork forces it to the CPU. See "Reproducing an
A1111 render" above — and note that a wrong choice here is invisible under `euler`.

**This is also why `schedule.zig` computes in f32.** The tree's time axis is quantised to
1e-6 while an f32 sigma near 10 has an ulp of 9.5e-7 — the same order — so a schedule
value one ulp off can land in the neighbouring cell and that step draws unrelated noise.
Measured: one sigma of twenty crossing one cell costs **28.9 dB** (ComfyUI against
itself), staying inside it costs 0.6 dB. A quantised key has to agree digit for digit, so
here — uniquely — being *more* accurate than the reference is just being different.

- ⚠️ **`brownian.round6` is Python's `round(x, 6)`, and neither shortcut works.**
  `@round(x*1e6)/1e6` double-rounds; Zig's own formatter rounds the *shortest
  round-trip decimal* half-up, where CPython rounds the **exact** binary value
  ties-to-even. So it is done exactly (integer mantissa arithmetic, then a
  correctly-rounded parse). Ties are reachable in practice — an interval midpoint that
  is an odd multiple of 1/128 has exactly 7 decimals ending in 5.
- ⚠️ **The sort-sign negation is real.** `BatchedBrownianTree` multiplies by the product
  of its construction-order and query-order signs; sampling runs *down* the sigma axis
  while the tree was built over `[sigma_min, sigma_max]`, so every increment is
  **negated**. Statistically invisible, bit-exactly essential.
- ⚠️ **The "bounce up to the parent" branch in `_loc_inner` is load-bearing, not
  defensive.** In halfway mode a split cuts at the node's *midpoint*, not at the
  requested boundary, so the child the walk descends into is routinely too narrow for
  the query; the reference bounces back to the parent, which by then has a midpoint and
  takes the straddle path. Omitting it walks the right edge forever. (First bug here.)
- ⚠️ **The tree span comes from the schedule BEFORE the first-sigma offset**, and Levy
  area is never needed (`levy_area_approximation='none'`), so only the `W` branch of
  `_increment_and_space_time_levy_area` is ported — with its f32 operation order
  preserved, since a Python float times an f32 tensor is computed in f32.

**Verified, bottom to top** — this stack is only as good as its weakest link, so each
one is pinned separately:

| layer | check |
|---|---|
| `SeedSequence` | **bit-exact** vs numpy 2.2.6 (pool *and* derived state, spawned and unspawned) |
| Brownian tree | **bit-exact** vs `torchsde` through ComfyUI's own `BrownianTreeNoiseSampler`, incl. out-of-order and multi-node queries |
| the solver | whole trajectories vs ComfyUI's **own** `sample_dpmpp_2m_sde{,_heun}` over a toy analytic denoiser, 6 arms (both families, heun + midpoint, eta 0/0.5/1), rel 1e-4 |
| the schedule | vs ComfyUI's own `normal_scheduler` (see the SD section) |
| **in a real render** | the per-step noise vectors are **identical element-for-element** to ComfyUI's on a 512² SD1.5 render, all 9 steps |

`tools/gen_sampler_fixtures.py` generates the first, second and fourth by driving
ComfyUI/torchsde directly. The toy denoiser is `(x + c) / (1 + sigma)` on purpose: pure
f32 add/divide, no transcendental, so a trajectory mismatch is the solver's and not
libm's.

⚠️ **A pixel-level SDE match to ComfyUI is not attainable through this engine's UNet,
and that is a property of the sampler, not a defect here.** Measured on SD1.5 512²/10
steps against ComfyUI, and the two columns are the point:

| | ComfyUI vs **itself**, model output shifted by a fixed rel 1.3e-4 | TP vs ComfyUI (real UNet, which differs by rel 1.3e-4 at step 0) |
|---|---|---|
| euler | 64.0 dB | 36.4 dB |
| dpmpp_2m_sde_heun, eta=0 | 56.5 dB (−7.5) | 28.3 dB (−8.1) |
| dpmpp_2m_sde_heun, eta=1 | 37.1 dB (−26.9) | 19.0 dB (−17.4) |

The *deltas* agree: this sampler intrinsically amplifies any model-level disagreement
far more than Euler does (ComfyUI loses 27 dB against itself from a 1.3e-4 shift at
eta=1), and TP's penalty is inside that envelope. Two dead ends recorded so the next
person does not re-walk them: the gap is **not** the schedule (fixed, and the 10-step
schedule is bit-exact), **not** the noise (identical per step in a real render), and
**not** a latent-layout permutation (per-element agreement at every step). The first
amplification test used *independent* per-call noise and misleadingly showed no
amplification at all (45.5 vs 46.7 dB) — a systematic fixed-direction shift is the
faithful proxy for an implementation difference, and it is what produced the table.

## Image metrics and LPIPS

`tp.image` carries the comparison metrics (`mse`, `psnr`, `ssim`, `detailEnergy`) and
`tp.models.lpips` the perceptual one. Both are pinned against external references by
`tools/gen_lpips_fixtures.py` (torch + torchvision + the `lpips` pip package + skimage;
run it with an env that has them, e.g. `/home/qt/genai/ai-toolkit/venv/bin/python`),
which emits three fixture files — one per module it validates:

| fixture | validates | reference |
|---|---|---|
| `src/ops/assets/conv_fixtures.json` | `ops.conv` conv2d / maxPool2d | `torch.nn.functional` |
| `src/core/assets/image_metric_fixtures.json` | `image.mse/psnr/ssim` | numpy + `skimage.metrics` |
| `src/models/assets/lpips_fixtures.json` | `models.lpips` | the `lpips` pip package |

- **LPIPS is one specific computation, not "a perceptual distance"** — see the module doc
  in `src/models/lpips.zig`. Measured agreement with the reference: **9e-9 absolute** on the
  fixture pairs, with per-slice feature checksums so a regression localizes to a conv.
- **The weights (~10 MB) are a user-supplied checkpoint**, at
  `models/lpips/lpips_alex.safetensors`; the LPIPS tests self-skip without them
  (`-Dintegration` plus `requireModelFile`). Regenerate with the same script.
- **SSIM's conventions are a real trap** and both live behind `image.SsimWindow`:
  `.uniform7` is scikit-image's default (7x7 uniform, sample covariance) and `.gaussian11`
  is what Wang et al. specify. They are different numbers; a caller has to choose.
- `ops.conv` is the *general* channel-last conv (any kernel/stride/padding, banded
  `im2col` + the shared GEMM). `wan_vae.conv2d` predates it and stays as it is because the
  VAE parity fixtures pin its bytes; a test asserts the two agree bit-for-bit on the k=3
  same-padding shape they share. ⚠️ The banding cap is **not** bit-neutral — a band is one
  GEMM and the CPU GEMM's reduction order depends on its row count, so the cap is a fixed
  constant (`conv.default_band_bytes`) rather than a tuning knob.

## Zig 0.16 conventions (differ from older Zig — do not use pre-0.16 patterns)

- `main` takes `std.process.Init`: `pub fn main(init: std.process.Init) !void`. Get the process-lifetime allocator via `init.arena.allocator()`, args via `init.minimal.args.toSlice(arena)`, and the `Io` instance via `init.io`.
- I/O goes through `std.Io`: writers are `*Io.Writer`; stdout is set up as `Io.File.Writer.init(.stdout(), io, &buffer)` and must be explicitly `flush()`ed.
- Container types are unmanaged-style: e.g. `std.ArrayList(T)` is initialized with `.empty` and takes the allocator per call (`list.append(gpa, x)`, `list.deinit(gpa)`).
- Fuzz tests use `std.testing.fuzz` with a `*std.testing.Smith` input generator.

## Dependencies

Runtime `dlopen`'d system libraries, gated per backend:
- Vulkan loader (`libvulkan.so.1`) — `--backend vulkan`.
- CUDA driver (`libcuda.so.1`) — `--backend zig-cuda` (hand-emitted PTX).
- `--backend cuda`: `dlopen`s NVIDIA's closed-source math libraries `libcublasLt.so` (int8/f16 GEMM) and `libcudnn.so.9` (fused SDPA attention + conv).
