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
  restore every shape check downstream fails on a perfectly well-formed file. The
  restore is skipped (with a warning) for block-quantized dtypes, whose on-disk
  layout follows the *stored* row length — re-labelling those rows would move every
  block boundary.
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
