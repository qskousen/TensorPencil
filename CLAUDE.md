# CLAUDE.md

## Ground Rules

- Talk through what you are doing: a short sentence on what and why, rather than a dozen silent tool calls.
- If you notice a problem, or a comment that is extraneous or too wordy, in a file you are editing, fix it.
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
- **Default to `zig build test` (fast, ~15s CPU unit suite). Do NOT run `zig build test -Dintegration` unless you truly need it** — it runs the GPU device tests and real-model inference tests and takes ~17 minutes. Reach for `-Dintegration` only when your change touches GPU kernels / device code or the real-model LLM/parity paths, and even then prefer narrowing with `-Dtest-filter="<substring>"`.
- If the user asks for something that may cause issues, push back and get confirmation before doing it.
- If you see existing code that may cause issues or is Band-Aid patch code, call it out and suggest a fix.
- There's no risk to trying big complicated work. We want to try unusual things. Be bold and adventerous.
- However, bold is not the same as sprawling: keep it structured and organized, and generalize where generalizing is cheap.
- **Cross-platform code; don't lock ourselves into Linux-only.** Even where a subsystem currently only runs on Linux (e.g. the CUDA/NVIDIA backend), reach for portable std APIs (`std.Io` futex/mutex/sleep, `std.posix`, `std.Thread`) over raw Linux syscalls (`std.os.linux.*`) unless there's a real reason none of them fit — so a future macOS/Windows port isn't blocked by avoidable platform lock-in. If you must go platform-specific, gate it behind a comptime `builtin.os.tag` branch with a portable fallback and call it out. `zig build -Dtarget=x86_64-windows` builds every exe for Windows, libav and libvips included (from the pinned `*_win` packages, DLLs installed beside the exes), and `zig build test` there compiles every test binary but cannot run them; `tools/sync_windows.sh <user@host>` mirrors the working tree to a Windows box over OpenSSH for a native build and test run. `core/filemap.zig` (whole-file mapping), `core/dynlib.zig` (runtime library loading, which `std.DynLib` has no Windows arm for), `core/diskspace.zig` (free space, which std wraps nowhere) and `core/rawtty.zig` (raw keyboard input: termios or console modes) are where the per-platform branches live; a new one belongs there rather than at a call site.
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
| `VIDEO_PLAN.md` | the video/audio roadmap, MiniMax H3 first. Its "silent wrong answers" list is the reason to read it before touching any of that code |
| `DIFFKEEP.md`, `DIFFKEEP_INTEGRATION.md` | the embedding encoders for the DiffKeep consumer |
| `VULKAN_MEMORY.md` | Vulkan subgroup/shared-memory rework notes |
| `TODO.md` | open work |

## Commands

- `zig build` — build the executables (`zig-out/bin/TensorPencil`, `zig-out/bin/tp-llm`)
- `zig build run -- <args>` — the diffusion CLI
- `zig build run-llm -- <args>` — `tp-llm`, the LLM CLI
- `zig build run-gui -- <args>` — `tp-gui`, the desktop GUI; `zig build gui` only compiles it
- `zig build serve` — `tp-serve`, the engine host daemon tp-gui spawns beside itself
- `zig build -Doptimize=ReleaseFast` — required for any timing measurement.
- `zig build test` — fast CPU unit suite (~15s). Integration tests are gated OFF.
- `zig build test -Dintegration` — everything, including GPU device tests and real-model
  parity tests (~17 min on the 3090). Individual tests self-skip when their device or file
  is absent. It defaults the build to ReleaseSafe (an explicit `-Doptimize=` wins) and runs
  the test binaries one at a time.
- `zig build test -Dtest-filter="<substring>"` — one test. **The `=` is required**, only
  one substring per run, and a filtered green run can contain no test at all: prove it ran.
- `zig build test -Dintegration -Dtest-timing` — per-test wall time, slowest first. Reach
  for it before "the tests are slow".
- `-Dmodel-root=<dir>` — where real-model tests and device-command defaults find
  checkpoints (default `/mnt/watch/models/diffusion`: `checkpoints/`, `vae/`,
  `text_encoders/` ...).
- `zig build gui-test` — tp-gui unit tests (`shared`, `engine`, `client`, and the gui files
  with arithmetic in them). Not part of `test`. **A file's tests only run if a step compiles
  it**; a new test file goes in its root's `test {}` block.
- `zig build test -Dtarget=x86_64-windows` — compiles every test binary for Windows.

Probes, for seeing what the code does without a debugger. Each prints its own usage.

- `ui-probe` renders tp-gui screens to a PNG from canned data, no engine or GPU. The GUI's
  failure modes are visual; this is how you see them.
- `catalog-probe <folder>...` prints what a model scan took each file for ("why is my
  model not in the menu").
- `driver-probe`, `serve-probe`, `hosts-probe` drive the engine host in-process, through a
  spawned tp-serve, and across two hosts; `tls-spike` checks the remote link with no
  engine. Always hand them a scratch copy of the config, never the user's.
- `chat-probe` is where chat template work happens (build with `-Dprivacy=false`).

**The daemon keeps none of what the user types.** `serve/privacy_test.zig` is the check,
and `src/serve_main.zig`'s module doc lists every path tp-serve may open for writing; diff
it against `serve-probe` under `strace -ff -e trace=openat,creat,rename,unlinkat` after
touching anything that writes. Paths that print a prompt or reply are compiled out under
`-Dprivacy` (default on). A flag is a promise; a missing code path is a fact.

## Testing and validation

The integration gate is `src/test_gate.zig`: GPU `init` fails in test builds without
`-Dintegration`, and heavy tests call `test_gate.requireModelFile` / `requireIntegration`.
Gate new slow tests the same way; keep fast CPU unit tests ungated. A SECOND gate is the
`testdata/gpu-tests` marker file (git-ignored): without it the device tests skip, and
`-Dintegration` reports green having run almost none of them. Two hazards that pass a test
alone and fail it in the full binary: both device weight caches key on the HOST POINTER (a
freed buffer's address is reused and served the previous upload, so every weight in a test
lives to the end), and `chat.applyTokenizer` publishes special ids process-wide (restore
with `chat.tokenizerIds`).

**Device validation is CLI commands, not unit tests**, because the test binary brings up no
CUDA context. Each family and each shared dispatcher has a `*-cuda-test` / `*-vk-test` (or
`lin-cuda-test`, `lora-cuda-test`, `wan-vae-test`, `te-test`, ...) that checks kernels
against their CPU ops and then a whole forward against the CPU forward, exiting non-zero on
failure; BACKEND.md's `Gates:` lines name them per family, and `main.zig`'s dispatch is the
full list. The `*-bench` commands time kernels and forwards.

- **Every `*-test` command runs its CPU side with `ops.matmul.exact_activations`**: below
  16 rows the CPU block-quant GEMM otherwise quantizes the activation too, which made a
  reference 3% wrong. Each takes `--act f32|f16|bf16`, the width its device forwards store
  activations at.
- **A device check pinned to one shape answers about one shape.** The diffusion device
  commands take the axes a defect rides (`--layers`, `--lat`, `--cap`, `--sigma`), and
  `--dequant-at-load` runs a quantized file through the dense path.
- ⚠️ **This box's GPU is shared with the desktop and its clock drifts ~3% between runs.**
  Measure a speed change with an INTERLEAVED same-binary A/B (both variants alternating in
  one process), never two invocations; check `nvidia-smi` for other users first.
- **A quantized checkpoint that renders is not a quantized checkpoint that works.** A lost
  conditioning renders a clean picture that ignores the prompt, so check a quantized file
  against its dense source on an INTERMEDIATE: `cond-diff <dense> <quant>` (conditioning),
  `velocity-diff <ckpt>` (device denoiser against CPU), and `quant-attrib <dense> <quant>`
  or `weights.Overlay` to attribute the damage to a layer or weight kind.
- **Judge an activation dtype** by `diffusion-profile --ab act:D` against `--ab perturb`
  (the same forward on a nudged latent) and against the CPU (`velocity-diff --dit-act`):
  W8A8 routes and 16-bit accumulators re-roll their own noise on any input change, so
  f32-vs-16 alone reads that noise as the dtype's cost.
- ⚠️ **A solid white image with no error is f16 overflow** (65504) somewhere in the chain;
  it has happened in five places. Instrument non-finite values and magnitudes per stage.
  `auto` activation precision finite-checks each forward and widens f16 to f32.

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
  `tools/render_sd_ref.py` renders through ComfyUI's own sampler for a whole-render A/B.
- **Read the control row first.** These models disagree with themselves across dtypes by
  more than they disagree with us; a PSNR figure without its precision floor is
  uninterpretable.
- **A green summary is not coverage** — gated tests self-skip when their checkpoint is
  missing. Check that a test actually ran before citing it.
- **Layout permutations are rms-preserving.** Planar vs channel-last and the per-family
  patch orders all keep every norm and magnitude when wrong; only the image differs. Pin
  each direction with its own test.
- AGPL references (A1111) are **fetched at generation time, never vendored**; the fixture
  records the upstream sha256.

## Architecture

Layered modules, wired in `build.zig`; each is independently importable so a consumer can
depend on one tier. `LIBRARY.md` has the detail.

| module | root | holds |
|---|---|---|
| `tp_core` | `src/core/core.zig` | tensors, dtypes, containers (safetensors/GGUF), tokenizers, samplers, schedules, RNG, image, and the per-platform shims |
| `tp_ops` | `src/ops.zig` | CPU numeric kernels (GEMM, attention, conv, norms, quant decode) |
| `tp_gpu` | `src/gpu.zig` | Vulkan (Zig→SPIR-V) and CUDA (driver-API PTX + cuBLASLt/cuDNN) backends |
| `tp_runtime` | `src/runtime/runtime.zig` | VRAM arbiter, residency planner, stepper `boundary` hook; pure std |
| `tp_models` | `src/tp_models.zig` | model architectures (`src/models/`) and LLM generation (`src/llm/`) |
| `TensorPencil` | `src/root.zig` | the umbrella module; anything public must be re-exported here |

Executables are thin drivers: `src/main.zig` (diffusion CLI and every device check),
`src/llm_main.zig`, `src/gui_main.zig` (`src/gui/`), `src/serve_main.zig` (`tp-serve`).

**The app is four layers, each a build module** (`build.zig` `layers`): `src/shared/`
(settings, catalog, model spec: what tp-gui and tp-serve both hold), `src/engine/` (the chat
session, the diffusion engine and `driver.zig` / `host.zig`: what owns a GPU), `src/client/`
(the client's mirror of each host, placement, transfers) and `src/gui/` (dvui views only).
`src/serve/` is the wire protocol, pure std. **tp-gui runs no engine and links no `engine`
module**: it talks to one or more tp-serve hosts, local or remote. Start from the module
docs of `client/hosts.zig` (hosts, dispatch, reconnects), `client/sched.zig` (placement),
`engine/driver.zig` (threads and lock order) and `serve/blob.zig` (model transfer).
Cross-layer imports are `@import("shared").config`, never a relative path. `catalog-probe`,
`chat-probe` and `driver-probe` link `engine` and `shared` with no dvui, which keeps those
layers free of it.

**The GUI has one design system, `src/gui/style.zig`**: a view never names a color, a font
face or a radius directly, and any user- or model-visible string goes through `fonts.zig`'s
run splitter (`addStyled` / `richLabel`) or it renders tofu outside Latin.

Rules for model and kernel code, each of which a reviewer will hold you to:

- A model architecture is normally `foo.zig` (CPU reference and loader), `foo_gpu.zig`
  (Vulkan), `foo_cuda.zig` (both CUDA backends) and `foo_dev.zig`, the device forward
  written ONCE over a per-backend `Dev` adapter. Read the architecture file's module header
  before touching it: it lists the conventions that are silent wrong answers.
- **A device forward names no weight dtype.** Every linear goes through
  `models/dual_lin.zig`'s routing and `lin_cuda.zig` / `lin_gpu.zig`'s execution; every
  convolution through `conv_cuda.zig` / `conv_gpu.zig`. A kernel wired in the dispatcher
  reaches every architecture; a kernel wired in one model is a regression.
- **A plain elementwise kernel is written once**, in `src/gpu/kernels/dual.zig`, which
  compiles to both SPIR-V and PTX. Only kernels needing shared memory or a subgroup reduce
  live per backend.
- **Every Vulkan shader module goes through `Context.createModule` and every dispatch
  through `Context.bind`**: that is where device limits and per-device patches apply. Pick
  a kernel's size from what the device reports, never from a vendor name.
- **Every format ComfyUI's quantizers emit reaches every family they support**: a new
  reader goes in `models/quant_weight.zig` the day it is written. A support scan reads
  every device linear (`device_lins`), never one tensor of one block.
- **A LoRA is a runtime sidecar, never a merge** (`models/lora*.zig`), on every family.
- **Cross-platform**: the per-platform branches live in `core/filemap.zig`,
  `core/dynlib.zig`, `core/diskspace.zig` and `core/rawtty.zig`.

## The diffusion pipeline

`pipeline.Session.generate` is composed from four public stages, and a caller can drive
them directly (img2img, inpainting, custom samplers, per-step measurement):

```zig
var cond = try sess.encode(gpa, prompt, .{});            // text -> conditioning
const sigmas = try pipeline.schedule(gpa, steps, shift); // steps -> sigma schedule
var den = try sess.denoiser(gpa, cond, null, 1.0, lat_h, lat_w, sigmas);
try den.predict(gpa, v, x, sigmas[i], null);             // one denoiser forward
var img = try sess.decode(x, lat_h, lat_w, .{}, null);   // latent -> RGB8
```

**`generate` composed from the stages must be bit-identical to `Session.generate`**; a gated
test compares the bytes. If they disagree, every measurement taken through the stages
describes a model nobody renders with.

`pipeline.Family` is one `Session` over every family, detected from the denoiser's own
tensor names (`detectFamily`), never from a flag.

| family | denoiser | text encoder | VAE |
|---|---|---|---|
| `krea2` | `models/dit.zig` | Qwen3-VL 4B | Wan 2.1 |
| `sd15` / `sdxl` | `models/sd_unet.zig` | CLIP-L (+ CLIP-G) | AutoencoderKL |
| `zimage` | `models/zimage.zig` | Qwen3-4B | AutoencoderKL (Flux) |
| `anima` | `models/anima.zig` | Qwen3-0.6B + T5 | Wan 2.1 |
| `sensenova` | `models/sensenova.zig` | the trunk's own base copy | none (pixel space) |
| `mageflow` | `models/mageflow.zig` | Qwen3-VL 4B | Mage-VAE or Flux 2's |
| `moss_image` | `models/moss_image.zig` | Qwen3.5 9B (HF) | Mage-Flow's |
| `qwen_image21` | `models/qwen_image21.zig` | Qwen3-VL 8B | Wan 2.2 layout, RGBA |

Weight formats (`models/quant_weight.zig` reads them all; BACKEND.md §7 has the matrix):

| format | storage | compute |
|---|---|---|
| bf16 / f8_e4m3 | dense | native on all backends |
| int8 convrot / tensorwise | I8 + per-row or one scale | int8 tensor-core GEMM (W8A8) |
| int4 convrot | nibble-packed + per-row scale | W4A4 on CUDA; Vulkan decodes to int8 |
| `asym_w4a8_int8` / `w6a8_int8` | 4- or 6-bit codes + fp8 group scales | decodes to int8 convrot per GEMM |
| NVFP4 | E2M1 + fp8 block scales | decodes to bf16 |
| GGUF block quants | ggml blocks | decodes per GEMM to int8 or a 16-bit dense (`--dit-gguf-gemm`) |

## Compatibility and metadata

- Sampling has four axes, each on the CLI and in the GUI: scheduler (`core/schedule.zig`),
  sampler (`core/sampler.zig`), prompt syntax (`core/prompt_*.zig`) and RNG compat
  (`core/noise.zig`). **A compat or dialect choice must reach every consumer of it**:
  wiring the initial latent but not the Brownian tree makes euler match while every SDE
  render is wrong.
- **The saved PNG records what the render USED**, in A1111's `parameters` block under
  A1111's and Civitai's own field names, never invented ones (`pipeline.appendExtraParams`).
  A reader re-renders from that block, so a hardcoded field there is a wrong answer.
- Where ComfyUI is not the right reference, the family's module doc says so (Qwen-Image
  2.1 samples on diffusers' schedule; MOSS-Image's ComfyUI render uses int8 attention).

## Checkpoints

- **`pipeline.Container.open` picks the reader by magic, not extension.** Any family ships
  bundled or split; `resolveComponent` decides per component, and an explicit
  `--text-encoder` / `--vae` flag wins.
- `weights.WeightStore` arms `Overlay` and `Mix` substitute tensors or whole layers from a
  second file without rewriting a checkpoint: the tools for attributing quantization damage.

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

This file is read in full at the start of every session. It orients an agent and states
how work is done here; it is not a description of how the code works (read the code and its
module docs for that) and not a record of what happened.

- **Write orientation and process only**: where something lives, how a change is checked, a
  rule every change must follow, a trap that is a silent wrong answer across the codebase.
  One or two sentences.
- **How one file or subsystem works goes in that file's module doc comment**, a backend
  capability in `BACKEND.md`, a Zig gotcha in `ZIG.md`, open work in `TODO.md`.
- **No history.** No dates, no "landed", no "was X before", no postmortems, no measured
  tables. `git log -p CLAUDE.md` has all of it.
- **Prefer deleting to qualifying.** If a section no longer helps someone start work, remove
  it rather than annotating it.
- **Budget: keep this under ~300 lines.**
