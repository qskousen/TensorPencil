# CLAUDE.md

Orientation for Claude Code (claude.ai/code) working in this repository: the rules, the
layout, and where to look for detail. It is not a changelog — see "Keeping this file
useful" at the end.

## Ground Rules

- Never run `git add` or `git commit` unless directly requested.
- Don't bring up "this code is uncommitted"; don't worry about commits or checkpoints or anything like that.
- `zig build` produces no output on success; any output indicates a warning or error.
- **NEVER run a built binary from the cache directly (e.g. `./.zig-cache/o/*/<exe>` or any hardcoded/globbed cache path) — always launch through the `zig build <step> -- <args>` command.** The cache holds *multiple stale binaries* from earlier builds; a glob or copied path silently runs an old one that predates your edits, producing bogus results. `zig build` recompiles and runs the *current* source every time. This applies to benchmarks (`embed-bench`, `ggml-bench`, …), `run`, `run-llm`, `run-gui`, and any other exe.
- `zig build test` is likewise silent when everything passes. Tests must NOT print diagnostics on success — use `errdefer std.debug.print(...)` before the assert so values only print on failure. Any stderr from a *passing* test makes the runner print a misleading red `failed command:` line (see ZIG.md); if `zig build test --summary all` says `test success`, nothing failed — don't investigate that line.
- **Read `ZIG.md` before doing any work.** It documents Zig 0.16.0 breaking changes relevant to this codebase. When you encounter and resolve a new 0.16.0 change, add it to `ZIG.md`.
- The best code is usually the simplest code that still handles every edge case: clear separation of concerns, modular, testable. Getting there takes work; do the work.
- If there is ambiguity in a request, don't guess or assume; ask for clarification.
- When adding a new feature or fixing a bug, add unit / integration tests as appropriate.
- Make sure all tests are still passing after working on something. If they aren't, fix it - even if the test was previously broken.
- **Default to `zig build test` (fast, ~15s CPU unit suite). Do NOT run `zig build test -Dintegration` unless you truly need it** — it runs the GPU device tests and real-model inference tests and takes ~11 minutes. Reach for `-Dintegration` only when your change touches GPU kernels / device code or the real-model LLM/parity paths, and even then prefer narrowing with `-Dtest-filter="<substring>"`.
- If the user asks for something that may cause issues, push back and get confirmation before doing it.
- If you see existing code that may cause issues or is Band-Aid patch code, call it out and suggest a fix.
- There's no risk to trying big complicated work. We want to try unusual things. Be bold and adventerous.
- However, bold is not the same as sprawling: keep it structured and organized, and generalize where generalizing is cheap.
- **Cross-platform code; don't lock ourselves into Linux-only.** Even where a subsystem currently only runs on Linux (e.g. the CUDA/NVIDIA backend), reach for portable std APIs (`std.Io` futex/mutex/sleep, `std.posix`, `std.Thread`) over raw Linux syscalls (`std.os.linux.*`) unless there's a real reason none of them fit — so a future macOS/Windows port isn't blocked by avoidable platform lock-in. If you must go platform-specific, gate it behind a comptime `builtin.os.tag` branch with a portable fallback and call it out.
- After adding a new kernel feature like relo, supporting a new dtype like bf16 or qk_6 for a backend, or anything similar, check BACKEND.md and update it to reflect the current state.
- Performance is CRITICAL, and we need to do what it takes to get there - don't skip out and do something easier if the hard work is what is needed.
- **A negative/limiting conclusion requires a receipt.** Before claiming an optimization "isn't worth it," "won't help," "can't be done cleanly," or "is too fragile/expensive," you must have an ISOLATION measurement that removes exactly the component in question (e.g. disable the op and re-time) — not a proxy and not an assumption. State whether each claim is measured or assumed.
- **A result that contradicts a strong prior means the measurement is suspect, not the prior.** (A 3090 being "flat" on batched matmuls is physically implausible → verify the harness before concluding — stale binaries, contention, wrong build.)
- **Name shortcuts explicitly and default to the robust option.** If an approach trades robustness for effort, say so, state the robust alternative and its real cost, and lead with the robust one — don't silently pick the easy path. A real tooling limit gets a clean workaround that does the full job, never a fragile hack or a reduced-scope "halfway."
- Explain things simply and clearly using common terminology. Avoid claudisms.
- **A comment is for the person about to change the code in front of them.** They already know Zig and the domain. Write only what is useful to them, not visible in the code itself, or a footgun for the careless. Everything else is noise, including whatever you learned on the way there.
- Comments in code should likewise be simple, to the point, and brief. Never use markdown, emojis, em-dashes, or other claudisms in code comments. Hyphens are fine where they make sense.
- Before writing a multi-paragraph comment, try to write it as one sentence. If that works, that was the comment. Length usually means the thinking isn't finished.
- Comments are not a history; git is the history. Never reference dates, and never name .MD files (including this one), checkpoint filenames, or models on this box. Naming a sibling source file (`dit_gpu.zig`) or a fixture generator (`tools/gen_*.py`) is fine and useful.
- Especially, never use the word "invariant" in any context whatsoever, nor a stand-in for it ("the contract", "the guarantee", "the property"). Say what must be true, not that something must be true.

## Project

TensorPencil is a diffusion inference engine (text-to-image) plus an LLM inference engine
(`tp-llm`) and a desktop GUI (`tp-gui`), written in Zig, targeting **Zig 0.16.0**
(`minimum_zig_version` in build.zig.zon). The engine is also exported as a library.

**ComfyUI is the compatibility target.** When an implementation choice is not derivable
from a checkpoint, match what ComfyUI does.

## Companion documents

| file | holds |
|---|---|
| `ZIG.md` | Zig 0.16.0 breaking changes and gotchas. Read first; add to it. |
| `BACKEND.md` | backend × feature × dtype support grid. Update when you add a kernel or format. |
| `LIBRARY.md` | the layered module split, for external consumers |
| `PLAN.md`, `LLM_PLAN.md` | diffusion and LLM roadmaps |
| `DIFFKEEP.md`, `DIFFKEEP_INTEGRATION.md` | the embedding encoders for the DiffKeep consumer |
| `VULKAN_MEMORY.md` | Vulkan subgroup/shared-memory rework notes |
| `TODO.md` | open work |

## Commands

- `zig build` — build the executables (`zig-out/bin/TensorPencil`, `zig-out/bin/tp-llm`)
- `zig build run -- <args>` — the diffusion CLI
- `zig build run-llm -- <args>` — `tp-llm`, the LLM CLI
- `zig build run-gui -- <args>` — `tp-gui`, the desktop GUI
- `zig build test` — fast CPU unit suite (~15s). Integration tests are gated OFF.
- `zig build test -Dintegration` — everything, including GPU device tests and real-model
  parity tests (~11 min; needs a device and the `models/` checkpoints). Individual tests
  self-skip when their device or file is absent.
- `zig build test -Dtest-filter="<substring>"` — one test. **The `=` is required**, and
  only one filter substring per run.
- `zig build gui-test` — tp-gui config unit tests (not part of `test`)
- `zig build -Doptimize=ReleaseFast` — optimized; required for any timing measurement.

The gate lives in `src/test_gate.zig` (`build_options.integration`): GPU `init` fails in
test builds when it is off, and heavy tests call `test_gate.requireModelFile` /
`requireIntegration`. Gate new slow tests the same way; keep fast CPU unit tests ungated.

**Device validation is CLI commands, not unit tests**, because the test binary brings up no
CUDA context. Each checks kernels against their CPU ops and then a whole forward against the
CPU forward, exiting non-zero on failure: `sd-cuda-test`, `cuda-dit-test`, `cuda-vae-test`,
`zimage-cuda-test`, `anima-cuda-test`, `te-test`, plus the `*-bench` commands
(`anima-cuda-bench`, `anima-vk-bench`, `vk-norm-bench`, `zimage-cuda-bench`).

## Architecture

Layered modules, wired in `build.zig`; each is independently importable so a consumer can
depend on one tier. `LIBRARY.md` has the detail.

| module | root | holds |
|---|---|---|
| `tp_core` | `src/core/core.zig` | tensors, dtypes, containers (safetensors/GGUF), tokenizers, samplers, schedules, RNG, image |
| `tp_ops` | `src/ops.zig` | CPU numeric kernels (GEMM, attention, conv, norms, quant decode) |
| `tp_gpu` | `src/gpu.zig` | Vulkan (Zig→SPIR-V) and CUDA (driver-API PTX + cuBLASLt/cuDNN) backends |
| `tp_runtime` | `src/runtime/runtime.zig` | VRAM arbiter and residency planner; pure std |
| `tp_models` | `src/tp_models.zig` | model architectures (`src/models/`) and LLM generation (`src/llm/`) |
| `TensorPencil` | `src/root.zig` | the umbrella module; anything public must be re-exported here |

Executables are thin drivers: `src/main.zig` (diffusion CLI), `src/llm_main.zig`,
`src/gui_main.zig` (`src/gui/`).

A model architecture is normally three files: `foo.zig` (CPU reference and the loader),
`foo_gpu.zig` (Vulkan) and `foo_cuda.zig` (both CUDA backends, which share one code path
and differ only in whether GEMM/attention route to the vendor libraries).

## The diffusion pipeline

`pipeline.Session.generate` is composed from four public stages, and a caller can drive
them directly — img2img, inpainting, custom samplers, latent upscaling and per-step
measurement all need that:

```zig
var cond = try sess.encode(gpa, prompt, .{});            // text -> conditioning
const sigmas = try pipeline.schedule(gpa, steps, shift); // steps -> sigma schedule
var den = try sess.denoiser(gpa, cond, null, 1.0, lat_h, lat_w, sigmas);
try den.predict(gpa, v, x, sigmas[i], null);             // one denoiser forward
var img = try sess.decode(x, lat_h, lat_w, .{}, null);   // latent -> RGB8
```

- **`Denoiser` exists because `predict` cannot be a free function on the GPU backends.**
  Text fusion, rope tables, timestep vectors and the activation workspace are built once
  per image. It is bound to one resolution and borrows its conditionings, which must
  outlive it. `Session.predict` is the one-shot form.
- **`Session.decode` does not modify the caller's latent** (it denormalizes onto a copy).
- **`generate` composed from the stages must be bit-identical to `Session.generate`.**
  A gated test builds the same image both ways and compares bytes.
  If they disagree, every measurement taken through the stages describes a model nobody
  renders with.
- Each stage tags its VRAM (`encode` → `.te`, `denoiser` → `.latent`/`.dit`, `decode` →
  `.vae`) so the GUI meter stays meaningful.
- `Session.decodePlanar` is the one VAE-decode ladder for every family: whole-image → free
  VRAM and retry → GPU-tiled → CPU-tiled. A whole-image decode is a single tile covering
  the whole latent. `--vae-decode auto|whole|gpu_tiled|cpu_tiled` overrides it.

## Model families

`pipeline.Family` — one `Session` over all of them, dispatching internally, so every caller
of the stage API works on every family.

| family | denoiser | text encoder | VAE | latent ch |
|---|---|---|---|---|
| `krea2` | `models/dit.zig` (SingleStreamDiT) | Qwen3-VL 4B | Wan 2.1 | 16 |
| `sd15` | `models/sd_unet.zig` | CLIP-L | AutoencoderKL | 4 |
| `sdxl` | `models/sd_unet.zig` | CLIP-L + CLIP-G | AutoencoderKL | 4 |
| `zimage` | `models/zimage.zig` (NextDiT) | Qwen3-4B | AutoencoderKL (Flux) | 16 |
| `anima` | `models/anima.zig` (Cosmos-Predict2 + LLM adapter) | Qwen3-0.6B + T5 | Wan 2.1 | 16 |

- **The family is detected from the denoiser's own tensor names** (`detectFamily`), never
  from a flag. SDXL must be tested before SD1.5 (both are LDM UNets; `label_emb` is what
  distinguishes them), and Anima is identified by its LLM adapter, not its trunk, which it
  shares with stock Cosmos-Predict2.
- Family-aware surface, for things a caller cannot decide itself: `Session.family()`,
  `.schedule()`, `.latentChannels()`, `.scaleInitialNoise()`, `.parameterization()`,
  `.latentPreviewInto()`, `.denoiserStore()`, `.replaceDenoiser()`.
- Krea2, Z-Image and Anima are flow matching; the SD family is discrete-eps (sigma comes
  from a beta ladder, the model is conditioned on a timestep index, and the input is
  pre-scaled). The Euler step is shared because for eps-prediction the trajectory
  derivative *is* eps.
- Each architecture file's module header enumerates the conventions that are silent wrong
  answers when got wrong (patch feature order, modulation shape, norm epsilons, pad-token
  positions). Read it before touching one.

⚠️ **Layout permutations are rms-preserving.** The sampler works in planar `[c][h][w]`,
the UNets in channel-last `[h*w][c]`, and the patch orders differ between families and even
between a model's input and output. Every norm and magnitude still matches when these are
wrong — only the image is different. Pin each direction with its own test.

## Checkpoints

- **`pipeline.Container.open` picks the reader by magic, not extension** (safetensors or
  GGUF). Guessing wrong reports `InvalidHeader`, which says nothing about what happened.
- **Container style is orthogonal to architecture.** Any family ships either bundled (all
  components in one file under prefixes) or split. `resolveComponent` decides per component
  (`denoiser` / `conditioner` / `conditioner2` / `decoder`), in this order: an explicit
  `--text-encoder` / `--vae` flag wins, then the primary checkpoint's own copy under any
  known prefix, then a *defaulted* side path, else `error.ComponentNotInCheckpoint`. A
  defaulted path that does not exist is not an error; an explicit one that fails to open is.
- **`weights.WeightStore` has four arms**: safetensors, GGUF, `Prefixed` (a base plus a
  prefix, so no loader takes a prefix parameter) and `Overlay` (a base plus a name→view
  patch map, for substituting one tensor without rewriting a checkpoint). `Overlay.mapping()`
  returns null, since a patched tensor's bytes are outside the base mapping.
- `safetensors.initFromSlice` requires the tensor ranges to **cover** the payload exactly.
  Per-tensor bounds checks pass on a corrupt file; only the aggregate shows it.
- **GGUF**: dims are stored reversed, and `comfy.gguf.orig_shape.<name>` must be restored at
  parse time (converters reshape tensors whose contiguous dim is not a multiple of 256).
  Such tensors are flagged `flat_blocks`: values are fine, but rows are not block-aligned,
  so consumers needing row-aligned blocks materialize to f32.

## Weight formats

`models/quant_weight.zig` holds the **one** container reader for each quantized format, called
from every family's `mat`. ⚠️ **Every format ComfyUI's quantizers emit reaches every family
they support** — a new reader belongs in that shared module the day it is written.

| format | storage | compute |
|---|---|---|
| bf16 / f8_e4m3 | dense | native on all backends |
| int8 convrot | I8 + per-row scale, 256-wide rotation | int8 tensor-core GEMM (W8A8) |
| int4 convrot | nibble-packed + per-row scale | W4A4 on CUDA; Vulkan has no `sint4`, so it decodes per GEMM to int8 |
| `asym_w4a8_int8` | 4-bit codebook indices + fp8 per-group scales | decodes to int8 convrot, per GEMM |
| NVFP4 | E2M1 nibbles + fp8 block scales, swizzled | decodes to bf16, feeds the bf16 GEMM |
| GGUF block quants (q4_k, q6_k, …) | ggml blocks | **CPU only** — no GPU GEMM exists |

- Packed weights stay packed and each consumer decodes on demand; materializing at load
  gives back the memory the format saved.
- Quantization is **per weight, not per model**: a checkpoint may mix formats block by
  block. A support probe must scan every device linear, not one tensor of one block —
  `anima.deviceLins` is the single list all three Anima scans read.
- The activation prep is a property of the *activation*, the format a property of the
  *weight*; int8 and W4A8 share one prep.
- ⚠️ **Weight storage must gate every GEMM call site, not just the main one.** Fast paths
  that key on "not int8 and not bf16" happily read a packed 4-bit weight as fp8 bytes.
- ⚠️ **f16's 65504 ceiling is a real limit on real checkpoints**, met three times here
  (SDXL's VAE residual stream, the Flux/Z-Image VAE's attention logits, Z-Image's trunk
  activations). Symptom is a solid white image with no error. Fixes in use: bf16 instead of
  f16, an f32 scores plane, and `residual_act_div` (an exact power-of-two scale across the
  cast). `sd_vae.Config.act_f16` picks f16 activation storage per architecture, and it is
  **range** that gates it, not precision.

## Samplers, schedulers and prompt dialects

Four orthogonal axes, each selectable on the CLI and in the GUI, and each recorded in the
saved PNG's AUTOMATIC1111 `parameters` block (a reader re-renders from that block, so a
hardcoded field there is a wrong answer).

| axis | file | surface |
|---|---|---|
| where the steps go | `core/schedule.zig` — all 9 ComfyUI schedulers plus the family sigma tables | `--scheduler` |
| how to step | `core/sampler.zig` — euler, dpmpp_2m_sde, dpmpp_2m_sde_heun | `--sampler`, `--sde-eta`, `--sde-s-noise` |
| prompt syntax | `core/clip_tokenizer.zig` (comfy), `core/prompt_a1111.zig`, shared parser `core/prompt_weights.zig` | `--prompt-syntax`, `--emphasis` |
| sampling compat | `core/noise.zig` selects `torch_rng.zig` or `philox_rng.zig` | `--compat`, `--rng`, `--sgm-noise-mult`, `--quantize-t` |

- `Scheduler.defaultFor` keeps the per-family default, so "model default" is a real choice.
- **`ddim_uniform` and `beta` do not return `steps + 1` sigmas.** Take the step count from
  `sigmas.len - 1`.
- **The SDE samplers need a real Brownian tree**, not a `randn`: `core/seed_seq.zig`
  (numpy `SeedSequence`) + `core/brownian.zig` (torchsde's dyadic `BrownianInterval`) +
  a noise generator. It keys on the sigma **quantized to 1e-6**, so a schedule value one
  ulp out draws unrelated noise — which is why `schedule.zig` reproduces torch's f32
  rounding exactly rather than being more accurate than it.
- A compat/dialect choice must reach **every** consumer of it. Wiring the initial latent
  but not the Brownian tree makes euler reproduce perfectly while every SDE render is wrong.
- Prompt weights are a CLIP-only feature. The capability is declared on the encoder
  (`TextEncoder.supports_prompt_weights`), never inferred from a family name; an encoder
  that cannot apply them warns and encodes verbatim.

## Reference implementations and fixtures

`tools/gen_*.py` generate fixtures by **executing** the reference (ComfyUI, diffusers,
transformers, torchsde, A1111's own parser) rather than re-deriving it. Two tiers:

- **ungated** pure-op fixtures embedded per module (`src/{core,ops,models}/assets/`) —
  a module can only `@embedFile` under its own tree, so each owns its fixtures;
- **gated** real-checkpoint parity, tied to a file by sha256, behind `-Dintegration` +
  `requireModelFile`.

Rules that have repeatedly earned their keep:

- **Verify a fixture has teeth** by deliberately breaking the implementation and confirming
  it fails. A generator should assert that its corpus distinguishes the settings it pins.
- **Matching a reference exactly still leaves the choice of reference unvalidated.** A
  correct port of the wrong convention passes every test. Only a render comparison against
  the actual target catches it.
- **A reference is a piece of code and can be the thing that is wrong.** When structure
  matches pixel-for-pixel and only tone differs, suspect the harness's output mapping.
- **Read the control row first.** These models disagree with themselves across dtypes by
  more than they disagree with us; a PSNR figure without its precision floor is
  uninterpretable.
- **A green summary is not coverage** — gated tests self-skip when their checkpoint is
  missing. Check that a test actually ran before citing it.
- AGPL references (A1111) are **fetched at generation time, never vendored**; the fixture
  records the upstream sha256.

## Zig 0.16 conventions (differ from older Zig — do not use pre-0.16 patterns)

- `main` takes `std.process.Init`: `pub fn main(init: std.process.Init) !void`. Get the process-lifetime allocator via `init.arena.allocator()`, args via `init.minimal.args.toSlice(arena)`, and the `Io` instance via `init.io`.
- I/O goes through `std.Io`: writers are `*Io.Writer`; stdout is set up as `Io.File.Writer.init(.stdout(), io, &buffer)` and must be explicitly `flush()`ed.
- Container types are unmanaged-style: e.g. `std.ArrayList(T)` is initialized with `.empty` and takes the allocator per call (`list.append(gpa, x)`, `list.deinit(gpa)`).
- Fuzz tests use `std.testing.fuzz` with a `*std.testing.Smith` input generator.
- `.arena = arena` in a struct literal copies the arena's state before later fields
  allocate into it. Build every field into a local first, then construct.

## Dependencies

Runtime `dlopen`'d system libraries, gated per backend:
- Vulkan loader (`libvulkan.so.1`) — `--backend vulkan`.
- CUDA driver (`libcuda.so.1`) — `--backend zig-cuda` (hand-emitted PTX).
- `--backend cuda`: also `libcublasLt.so` (int8/f16 GEMM) and `libcudnn.so.9` (fused SDPA
  attention + conv).

ggml is an optional build dependency (`-Dggml`, default on): its CPU quant kernels back the
GGUF block-quant dequant and GEMV paths. With `-Dggml=false` those dtypes return
`error.QuantBackendUnavailable`.

## Keeping this file useful

This file is read in full at the start of every session. It orients an agent; it is not a
record of what happened.

- **Write durable facts only**: where something lives, what dispatches on what, an
  rule that must hold, a trap that is a silent wrong answer. One or two sentences.
- **No history.** No dates, no "landed", no "was X before", no postmortems, no measured
  tables, no PSNR figures, no "this cost a debugging cycle". `git log -p CLAUDE.md` has all
  of it if anyone needs it.
- **Put detail where it is used.** A fact about one file belongs in that file's module doc
  comment; a backend capability belongs in `BACKEND.md`; a Zig gotcha in `ZIG.md`; open work
  in `TODO.md`.
- **Prefer deleting to qualifying.** If a section no longer helps someone start work, remove
  it rather than annotating it.
- **Budget: keep this under ~300 lines.** If a section grows past a screen, that is the
  signal it belongs somewhere else.
