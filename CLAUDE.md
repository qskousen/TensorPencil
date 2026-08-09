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

## A safetensors file must COVER its payload, and not checking cost a white image

`safetensors.initFromSlice` now refuses a file whose tensor ranges do not account for
every payload byte (`error.IncompleteMetadata`), with the two byte counts in the message.

⚠️ **The reference implementation has always made this check and we did not, so a corrupt
checkpoint rendered a solid white image instead of failing.** Reported as "tried to run
with this model and got `reached unreachable code`" on
`anima_baseV10-INT8_CONVROT-MIXED.safetensors`. Its header declares **2,324,776,986** bytes
of tensors inside a **2,366,726,170**-byte payload — 41,949,184 unaccounted — and the
actual data is displaced by ~41.94 MB from where the header says, so every tensor past the
first ~28 read a neighbour's bytes. Python's `safetensors` refuses it outright
("incomplete metadata, file not fully covered"); being more permissive bought only silent
garbage.

- **Every per-tensor check passed**: `end <= payload_len`, and
  `end - start == storageBytes(shape)`. The corruption is only visible in the *aggregate*,
  which is why per-entry validation is not enough.
- ⚠️ **The drift is not even constant** (+41,944,320 then +41,944,576), so no
  "reload at an offset" recovery is possible — the file is unusable, not merely shifted.
- **Measured before enforcing: 326 of 327** `.safetensors` files in the user's collection
  cover their payload exactly; the one that does not is that file. So this is a targeted
  check, not a new class of rejection. Worth re-running that census before tightening any
  other container invariant.
- ⚠️ The header is also unpadded (`8 + header_len` is 6 mod 8), leaving every `F32`
  tensor 2-byte misaligned. Harmless to our byte-wise conversions, fatal to any reader
  that casts a mapped range to `[]const f32` — worth knowing before adding a zero-copy
  fast path.

**Generalizable:** a container check that the reference implementation performs and we skip
is not a "permissive reader", it is a deferred crash. The failure surfaced five frames deep
in `ops.matmul` on an assert about something else entirely.

## ComfyUI's `asym_w4a8_int8`: a 4-bit weight that DECODES to int8 convrot

Landed 2026-08-08 for `krea2CenterSemiraw_v10Int8-ASYM_W4A8_INT8.safetensors`. Added
to ComfyUI in `344b4398 Support asym w4a8_int (#15308)`; the arithmetic lives in
`comfy_kitchen/tensor/w4a8_int8.py` + `backends/{eager,triton}/w4a8_int8.py`. A layer
ships five tensors — `weight` (I8, **`[N, K/2]`**, two 4-bit indices per byte),
`weight_s_rel` (F8_E4M3, `[N, K/group_size]`), `weight_s_channel` (F32, `[N]`),
`weight_codebook` (F32, `[16]`, optional) and `comfy_quant` — and decodes as

```
q    = nibble(packed)                              // 0..15, even col = LOW nibble
lvl  = codebook[q]   (or  q - 8  with no codebook)
int8 = rint(clamp(lvl * s_rel[row, group], -127, 127))
```

**The whole reason this cost almost nothing is the last line: from there it is the
int8-convrot GEMM this engine already runs** — `row_scale` = `s_channel`, `convrot` =
256 — so `Weight.dtype` is `.i8` after load and **all four backends ran it on day one
with no new kernel**. ComfyUI does the same thing (`_dequant_int4_grouped_to_int8`,
then `int8_linear(..., convrot=True)`), which is what makes a correct decode
bit-identical to its result rather than merely close. `ops/w4a8.zig` owns the decode,
`dit.Loader.matW4A8` the container half.

**Four things that are silent wrong answers rather than errors:**

- ⚠️ **The stored shape is EXACTLY the int4-convrot signature**, so detection must come
  first and must key on `weight_s_rel`, not on dtype or shape. `Loader.mat`'s existing
  heuristic reads "I8 at `[rows, cols/2]`" as nibble-packed **signed** int4 times a
  per-row scale; W4A8's nibbles are **unsigned indices into a non-uniform codebook**.
  (It happens to fail today on the missing `_scale`, but that is an accident of naming,
  not a check.)
- ⚠️ **The codebook is NOT uniform, so this is not `.i4` with a finer scale.** The 16
  levels are Lloyd-Max-optimal for a Gaussian — ConvRot makes the rotated groups
  Gaussian — spaced 0.186 at the tails and 0.103 in the middle. All 224 layers of this
  checkpoint ship the *frozen* table (`_FIXED_LUT`), but it is read per tensor because a
  heavy-tailed one gets a fitted table (`_codebook_for`'s kurtosis gate).
- ⚠️ **The rounding is ties-to-EVEN.** torch's `.round()` and the Triton kernel's
  `libdevice.rint` both are; Zig's `@round` is half-away-from-zero. `s_rel` is fp8, i.e.
  a small mantissa times a power of two, so exact `.5` products are rare but not
  negligible — every checkpoint would carry a handful of off-by-one weights, visible to
  nothing. The `ties_to_even` fixture case uses a half-integer codebook so 8 of 16
  levels distinguish the two modes; without it the difference cannot be seen.
- ⚠️ **`group_size` is derived from `weight_s_rel`'s OWN shape and only then
  cross-checked against `comfy_quant`.** Trusting the JSON alone would let a stale value
  read every scale from the wrong group — a finite, plausible weight. A disagreement is
  an error. `convrot_groupsize` is *not* derivable and must be 256, since
  `ops.convrot` is a comptime 256 table and a radix-4 FWHT over four base-4 digits.

**The decode goes through a 4 KiB LUT, and that is what makes it affordable.** `s_rel`
is fp8, so there are only 256 possible group scales; one `[256][16] i8` table therefore
holds every value a tensor can decode to (`w4a8.Levels`, built once per weight, exact by
construction — its entries use the same f32 multiply-round-clamp). The per-element work
becomes one L1 load and one store, so a whole-model decode is bandwidth-bound rather
than arithmetic-bound: **12.2 GB decoded in ~2.0 s** fanned over `std.Thread`s (the
model loaders are synchronous and take no `Io`; a spawn failure just means the caller's
thread does that share).

### Measured

**Pinned against comfy_kitchen's own reference in two tiers**
(`tools/gen_w4a8_fixtures.py`): 6 synthetic cases exact on every int8 value (frozen LUT,
no-codebook, `group_size` 4/16/32, saturation, ties) plus the full
`dequantize_w4a8_int8_weight` at 2e-6 through our convrot; and one **real layer** of the
checkpoint, hashed input and output, living in `tp_models` because that tier is about the
*container* (`tp_ops` cannot reach `test_gate`). Both tiers were verified to have teeth
by breaking them.

⚠️ **A render comparison of two differently-quantized checkpoints CANNOT rank the
formats, and reading only that table would have got this backwards.** Every pair of the
four krea2 quantizations lands in the same 22–26 dB band, because at 20 steps a small
weight perturbation is amplified into a different-but-equally-plausible image — the
control is that **ComfyUI's own fp8 and int8 renders of this seed differ by 24.71 dB /
SSIM 0.865**. The isolation that does rank them is at the *weight* level: dequantize
each format back to f32 in the original basis and compare (mean rel L2 over 4 layers
spanning the trunk).

| | vs fp8¹ | **vs int8 (reference-free)** | file |
|---|---|---|---|
| int8 convrot | 2.95% | — | 14.13 GB |
| **w4a8** | **7.88%** | **7.31%** | **8.81 GB** |
| int4 convrot (W4A4) | 16.80% | 16.53% | 8.05 GB |

¹ fp8-e4m3 is itself quantized (~2–3% RMS), so that column is inflated by an error
common to all three rows; the ordering holds, the absolute figures do not.

**So W4A8's peer is int4, not int8** — it is a 4-bit format whose per-16-group scale and
optimal codebook recover **2.26x** of int4's weight error at essentially the same file
size, while sitting ~2.5x behind int8. The renders agree once read that way (1120x1680,
20 steps, seed 252469767172722, `cuda`, against TP's own int8 render as the anchor):
**w4a8 23.57 dB / SSIM 0.876 against int4's 17.89 dB / 0.694.** Neither is *broken* —
int4 produces a good poster with a different pose, which is what quantization divergence
looks like as opposed to damage.

### The weight stays PACKED: a device decode per GEMM

⚠️ **The first version decoded to int8 at LOAD, and that threw away the point of the
format.** It worked on all four backends with no kernel at all — the decode's output is an
ordinary int8-convrot weight — but a decoded krea2 is **12.2 GB against 6.1 GB packed**,
i.e. exactly int8's footprint, and unlike an int8 checkpoint's mmap'd views those are
*anonymous* allocations (measured maxrss 19.3 GB against int8's evictable page cache). A
4-bit format that occupies 8 bits buys only accuracy and a smaller download.

So the packed form is what stays resident, and each consumer decodes on demand — the CPU
GEMM per k-slice into its existing dequant panel (`matmul.dequantW4A8Slice`), each GPU
backend per GEMM into a transient device scratch. That is also what ComfyUI's own Triton
and CUDA backends do. `Weight.dtype` is `.w4a8` with the sidecars on `Weight.w4a8`.

| `cuda`, 1120x1680 / 20 steps / cfg 1 | s/step | DiT VRAM | host maxrss |
|---|---|---|---|
| int8 convrot — the accuracy control | 2.25 | 12056 MB | mmap'd |
| int4 convrot (W4A4) — the *size* control | **1.95** | **6054 MB** | mmap'd |
| W4A8 materialized at load | 2.09 | 12056 MB | 19.3 GB |
| W4A8 packed, first decode kernel | 2.36 | 7081 MB | 12.3 GB |
| **W4A8 packed, final** | **2.26** | **7081 MB** | **12.3 GB** |

**So on `cuda` W4A8 now runs at int8's speed (2.26 vs 2.25 s/step) on 41% less VRAM**, and
lands within 16% of int4's speed and 17% of its VRAM while its weights are **2.26x closer
to int8's**. The residual VRAM over int4 is the fp8 per-group scales (0.77 GB), which are
intrinsic to the format, plus one 100 MB scratch.

**Every decode path is bit-identical to the load-time materialization** — `PSNR inf` on
all three GPU backends against the pre-change renders — so none of this moved the image.

⚠️ **Both device kernels are ALSO pinned against `ops.w4a8.decode` on synthetic weights,
and that is the durable check.** The render comparison above stopped being reproducible
the moment the checkpoint left the disk; a gated test that needs no model file does not.
Both were verified to have teeth by breaking them.

### Two kernels, and both were first written ~50x off their ceiling

Neither is arithmetic — the `[256][16]` level table means a decode is pure byte lookup —
so both have a hard bandwidth ceiling of ~20 ms/step for the whole model, and both missed
it badly in their first form. ⚠️ **Judge these by achieved GB/s against the card's roof,
never by share of the step** — this is the third time this file records that rule.

- **CUDA (`w4a8_decode` PTX, shared by `cuda` and `zig-cuda`): 270 -> ~100 ms/step.** The
  first version was one thread per packed byte. The level lookups are *dependent* loads
  (the address comes from the loaded nibble), so one byte per thread gives a single memory
  chain per thread and is latency-bound, not bandwidth-bound. Four bytes per thread
  (`u32` in, `v2.u32` out, `prmt.b32` packing) puts eight independent chains in flight for
  the same traffic. ⚠️ That makes `group_size % 8 == 0` a requirement — a `u32` of packed
  bytes must lie inside one group so the group scale is loaded once — and
  `dit.w4a8SmallGroup` refuses the checkpoint by name rather than asserting. No file uses
  a smaller group; Vulkan's kernel is general over it.
- **Vulkan (`w4a8_decode_t`): 1757 ms/step, isolated with a new `w4a8` profile
  category.** Its GEMM reads a k-major weight, so this kernel also has to transpose — and
  a transpose has one unavoidably strided side, which at 32 lanes per warp is 32 separate
  sectors per load instruction. Paying that *per GEMM* rather than once at load is the
  whole 50x. ⚠️ **The fix was to stop transposing here at all**: both the packed bytes and
  the fp8 group scales now go through `weightBuffer`, i.e. they are byte-transposed once
  at load and cached exactly like every dense weight, and the per-GEMM decode is then
  coalesced on all three streams. It reads `a.data[t]` — the thread id *is* the source
  index. ⚠️ Its post-rewrite speed is **UNMEASURED**: the checkpoints were pruned off this
  box before the render could be repeated. Correctness is pinned by the device test, not
  by a render.
  - The `w4a8` profile category is kept for the reason it was added: folded into `matmul`
    this cost looked like a slower GEMM, and a per-GEMM pass with a known ceiling deserves
    its own row.

⚠️ **The device weight caches key on HOST POINTER, and that bit the test, not the model.**
Both kernel tests allocate per case; freeing a case's arrays let the allocator hand the
same address to the next one, which scored a stale cache hit and decoded the *previous*
case's weights — a real failure ("got 0, want -4") that looked like a group-size bug. The
tests hold every case in one arena now. A model never hits it (weights live in the model
arena for its lifetime), but anything that reuses a weight-shaped buffer does.

⚠️ **`opI8Prep` is shared, and that is the structural reason this cost so little.** W4A8's
activation prep *is* int8's — the "A8" is exactly that the activation stays 8-bit — so
only the weight's storage differs, one GEMM entry point per backend changed, and a
checkpoint that mixes int8 and W4A8 linears works because the dispatch is per weight
(`i8GemmW`) rather than per model.

⚠️ **A load-timing label had been backwards for every family, and it sent me after the
wrong component.** All five arms of `Session.init` load the denoiser first and take `t1`
after it, but the summary line read `"models loaded (encoder {t1-t0}, dit+vae {t2-t1})"` —
so the W4A8 arm's 2.0 s of read-and-decode was reported as `encoder 2.0s, dit+vae 0.3s`,
i.e. as a slow *encoder* and a suspiciously instant 12 GB decode. The numbers were right
and the names were not; same class as a mislabelled profiler bucket. Now
`denoiser … , encoder+vae …`.

⚠️ **Also fixed on the way past: `safetensors`' "real model headers" test guarded on ONE
of the three checkpoints it opens** and then opened the other two unconditionally, so
pruning any single file hard-failed a *fast-suite* test with a bare `FileNotFound` five
frames deep. Each block guards its own path now, and the test skips rather than passing
green when none is present.

## ComfyUI's NVFP4: 4-bit E2M1 floats, and a weight-only format below Blackwell

Landed 2026-08-08 for three checkpoints at once — krea2, Anima and Z-Image all ship one.
Reference: `comfy_kitchen/tensor/nvfp4.py` + `backends/eager/quantization.py`. A layer
carries `weight` (U8 `[rows, cols/2]`, two E2M1 codes per byte), `weight_scale`
(F8_E4M3, per-16-element block), `weight_scale_2` (F32 scalar) and `input_scale`
(F32 scalar, **Blackwell-only, unused here**), decoding as

```
total[row, blk] = weight_scale_2 * fp8(block_scale[row, blk])
value           = E2M1[nibble] * total[row, blk]
```

⚠️ **`MIN_SM_VERSION = (10, 0)` gates the NATIVE op, not loading — NVFP4 runs fine on an
Ampere card and I initially said it could not.** ComfyUI's `pick_operations` calls
`supports_nvfp4_compute`; when false it moves `nvfp4` from what its log calls "native ops"
to **"emulated ops"**, which sets `_full_precision_mm`, and every linear then dequantizes
the weight to the compute dtype per call before a normal GEMM. The 4-bit weight stays
resident. This engine does the same: `Weight.dtype == .nvfp4` keeps the packed bytes, the
CPU GEMM decodes into its f32 panel and each GPU backend into a transient f16 scratch
feeding the existing f16 tensor-core GEMM.

### Three conventions that differ from every other 4-bit format here

Each is a silent wrong answer, and each would be got wrong by analogy with `.i4`/`.w4a8`:

1. ⚠️ **Element 2k is the HIGH nibble** (`hi_first = True`). `.i4` convrot and `.w4a8` are
   both low-first. The wrong order permutes adjacent weight pairs, which is **exactly
   rms-preserving** — a fast test asserts both that it differs on 7584/8192 fixture values
   AND that the sum of squares is bit-identical, because that is why no magnitude check
   can find it.
2. ⚠️ **The block scales are SWIZZLED** into cuBLAS's tiled layout (`to_blocked`), while
   on disk they keep the LOGICAL `[rows, cols/16]` *shape* — so nothing in the header
   hints at it. `nvfp4.unswizzleScales` is the inverse of the reference's five reshapes,
   applied ONCE at load so no consumer carries a five-way tiled index in an inner loop.
   Measured: reading the stored order as row-major moves 6281/8192 values.
3. ⚠️ **The multiply association is `E2M1 * (per_tensor * block)`**, not
   `(E2M1 * block) * per_tensor`. Only the first is bit-exact against the reference, and
   it is what a per-scale-byte table computes naturally.

**No ConvRot and no per-output-row scale**, unlike int8/int4/W4A8 here — `row_scale` stays
null and `convrot` 0, and `matmul` asserts both are absent for `.nvfp4`. A `[256][16]`
table (`Levels`, f32 for the CPU panel and f16 for the GPU operand) folds the whole decode,
so the per-element work is one lookup.

⚠️ **A NaN block-scale byte PROPAGATES here**, where `ops/w4a8.zig` maps it to 0 — only
because W4A8's target is an integer and `@intFromFloat(nan)` is illegal, not because 0 is
more right. Both match their reference; the device tests assert NaN-for-NaN agreement.

### Detected by `weight_scale_2`, not by `comfy_quant`

⚠️ **Z-Image's NVFP4 checkpoint ships NO `comfy_quant` blob at all** (krea2's and Anima's
carry `{"format": "nvfp4"}`), so a loader keyed on the metadata silently fails to recognize
one of the three files. `_scale_2` is the tensor only this format has. `models/quant_weight.zig`
holds the ONE implementation, called from all three families' `mat` — three copies of
"read the scale, unswizzle it, build the table" is the drift that makes one family's
renders disagree with another's for reasons no shape check sees.

### Measured

**Pinned against comfy_kitchen in two tiers** (`tools/gen_nvfp4_fixtures.py`): 4 synthetic
cases **exact** at f32 output (the generator asserts our model of the reference before
emitting, and self-checks that both traps above change the answer), plus **one real layer
per family**, hashed input and output, through the actual loader. Both verified to have
teeth by flipping the nibble order — 3 tests fail.

| krea2, 1120x1680 / 20 steps / cfg 1 | s/step | DiT VRAM |
|---|---|---|
| **`cuda`** | **4.36** | **7034 MB** |
| `zig-cuda` | 6.39 | 8948 MB |
| `vulkan` | 6.52 | 7453 MB |
| `cpu` | 8.26 (256²) | — |

The three GPU arms agree with each other at **28.9–36.8 dB / SSIM 0.987–0.996**, and each
agrees with the CPU f32 decode at 256² (vulkan 28.5, cuda 23.7, zig-cuda 19.9 dB — a
4-step 256px trajectory is chaotic, so read the reference-resolution spread instead).

⚠️ **NVFP4 costs W4A8's VRAM at roughly TWICE W4A8's step time, and that is structural.**
Same checkpoint family, same settings: W4A8 is **2.26 s/step on 7081 MB**, NVFP4 **4.36 on
7034 MB**. Both keep 4-bit weights resident; the difference is the arithmetic partner —
W4A8's is the int8 tensor-core GEMM, NVFP4's is f16, at half int8's throughput and twice
the decoded weight traffic. **On this hardware NVFP4 is therefore the worse of the two for
a model that has both**, and its value is (a) running checkpoints you already have and
(b) being ready for Blackwell, where its native W4A4-fp4 GEMM would invert the comparison.

**All three families run the device trunk on all three GPU backends**, through one shared
loader and one `Meta`/`Levels` pair. At 256² / 4 steps, each device trunk against the SAME
model's CPU trunk — the check that matters for a port, since it is the same weights through
two independent decoders:

| device trunk vs the same model's CPU trunk, 256² / 4 steps | `cuda` | `zig-cuda` | `vulkan` |
|---|---|---|---|
| krea2 | 23.9 dB | 23.0 | 22.9 |
| Anima | 53.4 dB | 54.2 | 55.1 |
| Z-Image | 51.1 dB | 55.0 | 43.4 |

krea2's 23 dB is a 4-step 256px trajectory, which is chaotic; its reference-resolution
figure is in the table above. Anima's and Z-Image's 43-55 dB is the real signal: the same
weights through two independent decoders, differing only by bf16's rounding.

⚠️ **Do NOT take s/step from a 256²/4-step run on this card, and I did once.** Those
numbers put `cuda` SLOWEST — 2.2x behind the hand-PTX arm — which contradicts every other
table in BACKEND.md, and a contradicted prior means the measurement is suspect. Two causes,
and only the second was a real defect:
- **The 3090 idles its clocks**, so a short cold render is not a measurement: the *same*
  krea2 config read 0.36 and 0.66 s/step on consecutive runs. At 1120x1680 / 8 steps the
  repeat variance is ±1.3% and the ordering is the expected one (`cuda` **4.35-4.46**,
  `zig-cuda` 6.68, `vulkan` 6.56-6.60 s/step). This file already records the rule for
  `anima-vk-bench`; it applies to whole renders too.
- **`opMatmulNvfp4` forced the cuBLASLt arm through a padded staging buffer and a
  compaction pass it does not need.** cuBLASLt takes an arbitrary `m` and writes exactly
  `m` rows, so the libs arm now writes `y` directly — the same fast path `opGemmBf16`
  already had. Worth 4.73 -> 4.43 s/step at reference size, **183 MB of VRAM** (no
  `conv_c`), and 2 fewer kernels per GEMM. Verified bit-identical on Anima and Z-Image;
  krea2 moved 21.6 -> 23.9 dB against its CPU trunk, i.e. *closer*, because the staged path
  it replaced also had the `zeroBias` hazard below.
  - ⚠️ **`Backend.zeroBias` is not safe to hand to `cachedWeight`**: it reallocates on
    growth and returns the whole buffer, so its pointer MOVES, leaving a device buffer
    registered against a freed host address for a later allocation to collide with. The
    hand-PTX arm takes the caller's stable file-scope `zero_bias` instead — which is what
    `zimage_cuda.gemm` already documents for the same reason.
  - The hand-PTX arm still stages, because `launchHgemm` genuinely writes `mpad` rows; that
    is also why `zig-cuda` holds 9131 MB against `cuda`'s 7034.

⚠️ **Four things the per-family wiring turned up. Two were silent, two were fatal:**

- **f16's 65504 ceiling, and it made Z-Image a SOLID WHITE image.** The decoded weights are
  tiny, but the GEMM converts the *activation* to the same format, and Z-Image's trunk
  activations pass f16's range. The tell was that all three backends were **byte-identical**
  (same md5) at 7.50 dB — three independent GEMM implementations cannot agree pixel-for-pixel,
  so it was not arithmetic; white is white. Switching the whole format to **bf16** — the level
  table, both decode kernels' output and `coopF16WDispatch`'s `bf16_ab` — fixed it: **7.50 ->
  43-55 dB** on all three. ⚠️ **Named cost: bf16 is the ROBUST choice, not the free one.** Its
  8-bit mantissa is coarser than f16's 11, which costs krea2 ~1-2 dB and Anima ~4 dB at 256²
  against the f16 version — but a 4-bit payload cannot use f16's extra mantissa anyway, this is
  the same regime these models' own dense bf16 weights run in, and a per-model choice would be
  a knob that can be set wrong. Third time this repo has met f16's range on a real checkpoint
  (SDXL's VAE residual stream, the Flux/Z-Image VAE's 9.95e6 attention logits, now this).
- **`launchHgemm` writes `mpad` rows, not `m`** — `grid.y = mpad/128` and each block stores a
  whole 128x128 C tile — so it cannot write into a caller buffer sized `m*rows`. Reported as
  `CUDA_ERROR_ILLEGAL_ADDRESS` on Z-Image's first `[10240, 3840]` MLP weight at m=288/mpad=384,
  3.9 MB past the end. `opMatmulNvfp4` now stages into the padded `conv_c` and compacts, as
  `opGemmBf16` already did for exactly this reason. ⚠️ **`opMatmulFp8` still writes `y`
  directly** and so carries the same requirement on its callers implicitly; its zimage/anima
  `.f8_e4m3` arms have never been exercised and would hit this the day an fp8 checkpoint for
  either shows up.
  - ⚠️ Found by bisection, not by the sanitizer: `compute-sanitizer` on this box fails with
    "Unable to find injection library libsanitizer-collection.so". A temporary sync-and-report
    after each NVFP4 GEMM named the shape in one run.


- **Z-Image splits its fused qkv into three row-block GEMMs** (`qkvPart`), and a row range
  of an NVFP4 weight must slice the **per-block scales** with it — they are
  `[rows][cols/16]`, so it is a plain slice, but leaving the full array behind a shortened
  `Weight` makes the k and v views read **q's** block scales. `Meta.rowSlice` owns it and
  `qkvPart` now takes caller storage for the sliced `Meta`.
- **`zimage_*.supported` probed `layers[0].attn.qkv.dtype` — one tensor of one layer.**
  This file already flagged that as "correct only because no mixed Z-Image checkpoint
  exists"; it now scans every layer's linears (and each `ada`) against what the *device*
  has, and names the offending tensor. `anima.LinKind` gained `.nvfp4` as a distinct kind
  rather than folding into `.dense`, and the exhaustive switch that made the Vulkan arm
  fail to compile is exactly why that was the right shape.

⚠️ **The bug this cost, and it is the same lesson as the fp8-vs-GGUF blank image already in
this file.** `dit_gpu` has GEMM call sites BESIDES `Gemm.go` — the fp8 "shared" fast paths,
gated on `!is_i8 and !is_bf16`, which for a packed NVFP4 weight is **true**. So q/k/v/gate
and the MLP took `opMatmulCoopH16`, reading nibble pairs as e4m3 bytes. It was visible only
as **VRAM**: 84 layers cached at their LOGICAL byte count (2x packed), 8064 MB of an
11899 MB total, because a 4-step 256px render still looked plausible. Found by asking why
Vulkan held 11899 MB where CUDA held 6737 — a histogram of the weight cache showed 84
entries at exactly `rows*cols` bytes. Fixing it took Vulkan to **6869 MB and 2.4x faster**,
and turned it into the arm that agrees BEST with the CPU. **Weight storage has to gate
every arm, not the one you remembered writing.**

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

**Then Vulkan's scores plane got QUERY-BANDED**, which was its single largest item.
`sd_vae_gpu.attn` loops over bands of `sd_vae.scoresBand(seq)` queries; `attn_scores`
takes an offset + band size in `u5`/`u6` and `attn_out` an offset in `u6` **encoded
+1, so 0 keeps meaning "not banded"** — that kernel had exactly one free push slot and
had to distinguish "band starting at row 0" from "no band". Every pre-existing caller
passes 0, so the DiT, UNet and CLIP paths are unchanged *by construction*.
- The arithmetic is untouched because `attn_out` already runs an **online softmax over
  whole key rows**, so a band of queries is exactly independent work. Render verified
  **bit-identical** (PSNR ∞).
- ⚠️ `scoresBand` lives in `sd_vae.zig` and is called by BOTH the decoder and
  `estimatePeakBytes`. Two copies of that constant is precisely the drift that makes a
  peak estimate lie — the same duplication that had the GUI refusing GGUF encoders the
  engine loads fine.

**Vulkan then got the same f16 activation storage**, so both GPU backends now read
`Config.act_f16`. Six kernels rather than CUDA's five: SPIR-V storage buffers are all
`[*]f32`, so f16 rides as **u32 pairs** — element `i` is half `i & 1` of word
`i >> 1`.

- ⚠️ **Reads may be per-element; a WRITE must own the whole word**, or two threads
  race on it. So every store twin (`gn_apply_h16`, `add_h16`, `bias_compact_h16`)
  indexes PAIRS and does two elements per thread. `gn_apply_h16`'s pair can land in
  two different GroupNorm groups (`per_group = ch/32` is odd at ch = 96 or 160), so
  its group lookup is per element, not per pair.
- The sixth is `h16_to_h16_pad`: Vulkan's k=1 conv converts its activation *inside*
  `coopF16WDispatch` via `f32_to_h16_pad`, so an f16 source needs an f16-input twin
  of that. CUDA got this free — `f16_pad2d` already existed there.
- ⚠️ **A first attempt wrote all five against CUDA's push layout and was thrown
  away.** They compiled. Vulkan's `im2col_sd` carries the source width in `u3` and
  the output width in `u6`; `bias_compact` carries a bias offset in `u4` and
  `act_div` in `f0` — none of which CUDA's do. Two backends with same-named kernels
  and different push conventions is exactly the trap: **read the f32 original in the
  same file, never port the twin from the other backend.**

| backend | decode | est peak | note |
|---|---|---|---|
| **`cuda`** | 12.0 -> **1.4 s** | 7.7 -> **2.2 GB** | cuDNN SDPA + f16 storage |
| **`vulkan`** | 2.1 s | 7.7 -> **2.6 GB** | banded plane + f16 storage |
| **`zig-cuda`** | 13.2 -> **1.4 s** | 7.7 -> **2.2 GB** | f16 storage + f32 scores band |

Vulkan's f16 render is **68.54 dB / SSIM 0.9999** against its own f32 one (max
difference one 8-bit level) — matching CUDA's 68.52 dB, which is what you want to see
from two independent implementations of the same idea.

**Finally `zig-cuda` got an f32 SCORES BAND, and its decode went 13.2 -> 1.4 s**,
bit-identical. `opAttnTCFlash` stored its query band in f16, which this VAE's 9.95e6
logits cannot survive, so `sd_vae_cuda` had been falling back to the naive per-query
kernel. Three pieces, and two of them already existed:

- the scores GEMM is `hgemmBatchedFn` — `buildHgemm`'s `c_f16` was already a
  parameter, so the f32-C form was there all along;
- `softmax_md_f32` is **derived** from `softmax_md_f16` by `replaceOnce`, a comptime
  substitution that `@compileError`s if a pattern is absent **or not unique**. The
  two differ in three lines (element size and the load), and this way they cannot
  drift — it caught a wrong pattern of mine at compile time;
- `buildHgemm` gained `a_f32`: the `attnout` GEMM reads S as a pair of f32 instead of
  a packed f16 word. ⚠️ **P stays f16** — it is a probability in [0,1], so only S
  widens and the MMA is untouched. A needs its own row step and k offset because B
  keeps its f16 stride.

⚠️ **The routing gate was on the wrong axis, and that is the actual bug this
exposed.** `opAttnTC` sent single-head attention to the flash path only when the
plane exceeded the scratch budget — so a SMALL latent took the batched f16 path and
came out non-finite while a large one was fine. Size never decided correctness; score
magnitude did. Single-head now goes to the flash path at every size, which costs
nothing (it degenerates to one band when the plane already fits).

⚠️ **Measured dead end, recorded so nobody re-walks it:** before this, a warp-per-query
`attn` was tried — coalesced loads, `hd/32` accumulator instead of `hd`. It is **18.3x**
faster at seq 1736 and **1.1x** at seq 26,136, because at the real size both naive forms
sit at ~0.13 TFLOP/s and neither coalescing nor L1 reuse is what binds them. It was
reverted. The isolation that settled it is in `zimage-cuda-bench`:

| at seq 26,136, 1 head x 512 | time | TFLOP/s |
|---|---|---|
| `be.attn` (thread per query) | 11,431 ms | 0.12 |
| `opAttnTC` (flash, f32 scores) | **34 ms** | **40.85** |

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

⚠️ **Found, NOT caused: `cuda-dit-test` on a krea2 checkpoint fails at BOTH dtypes** —
rel L2 0.141 (bf16) and 0.132 (int8 convrot) against the CPU forward, the bf16 arm also
taking ~11.5 s for a 256px forward.
A/B'd across the `qkNorm` change (0.14061 with the old kernel, 0.14098 with the new), so
it is pre-existing and unrelated to any of the above. Unfixed, and worth someone's time:
it means krea2's bf16 CUDA arm has no working validation today.

## Anima: a fifth family, and the first Cosmos derivative

⚠️ **Anima IS Cosmos-Predict2's DiT.** `comfy/ldm/anima/model.py` is 214 lines that
subclass `MiniTrainDIT` from `comfy/ldm/cosmos/predict2.py` and bolt an `LLMAdapter`
onto its front; `model_detection` picks `"anima"` over `"cosmos_predict2"` purely
because `llm_adapter.blocks.0.cross_attn.q_proj.weight` exists. So everything in
`models/anima.zig` that reads as "a video DiT used on one frame" is exactly that —
`patch_temporal = 1`, `T = 1`, and a 3-axis RoPE whose temporal axis is all zeros.

Landed 2026-08-07 on the CPU and, later the same day, on **all four backends** (see "###
On the GPU" below). It cost less than the SD family did because most of it already existed;
what was genuinely new is one tokenizer and one 2-part model.

| piece | what it is | reuse |
|---|---|---|
| tokenizer A | Qwen2 BPE | `core/tokenizer.zig` + a new `encodeSegmented` |
| tokenizer B | **SentencePiece Unigram** (T5, 32100) | **new**: `core/t5_tokenizer.zig` |
| text encoder | Qwen3-**0.6B**, hidden 1024 | `qwen3.TextEncoder`, new `.anima` `Variant` |
| VAE | Wan 2.1, 16-ch, `Wan21` latent format | `wan_vae.Decoder` **verbatim**, krea2's arm |
| sigma table | `ModelSamplingDiscreteFlow`, shift 3.0 | `SigmaTable.discrete_flow` — **bit-identical to Z-Image's** |
| denoiser + adapter | 28 blocks + 6 adapter blocks | new: `models/anima.zig` |

⚠️ **The prompt is tokenized TWICE, by two unrelated tokenizers, and the adapter is
where the two streams meet.** The `llm_adapter`'s *queries* come from its own
`Embedding(32128, 1024)` indexed by **T5** ids; its *keys and values* are the
Qwen3-0.6B encoder's final hidden states. Its output — `max(512, n_t)` rows, zero
padded — is what the denoiser cross-attends to. So `Cond.data` here is the adapter's
output, not the encoder's, and `Cond.seq` is 512 for any prompt under 512 T5 tokens.

⚠️ **Emphasis weights live on the T5 branch ONLY, which makes Anima the first
Qwen3-conditioned family where `supportsPromptWeights` is TRUE.**
`AnimaTokenizer.tokenize_with_weights` forces every Qwen3 weight to 1.0 and keeps the
T5 ones, which then multiply the adapter's output ROWS — a plain per-row multiply, not
CLIP's `(z - z_empty)*w + z_empty` interpolation. krea2 and Z-Image are false for a
structural reason (no fixed token window); Anima has one, indexed 1:1 by the T5
tokenization.

### The seven conventions that are silent wrong answers

Enumerated in `models/anima.zig`'s header; the two worth repeating here:

1. ⚠️ **The input and output patch feature orders are DIFFERENT.** `x_embedder` takes
   `(c, ph, pw)` — channel **slowest**; `unpatchify` emits `(ph, pw, c)` — channel
   **fastest**. They are not each other's inverse. Using one for both is a pure
   permutation: every norm, every magnitude and every per-stage rel-L2 still matches,
   and only the image is wrong. Third time this repo has met that class of bug (SD's
   planar/channel-last, Z-Image's `(ph, pw, c)`), hence a test that pins each
   direction separately and asserts the two disagree.
2. ⚠️ **The timestep IS the sigma.** `sampling_settings` has `multiplier: 1.0`, so
   `model_sampling.timestep(sigma)` is the sigma itself, in (0, 1]. Z-Image — whose
   sigma table is **bit-identical** — feeds `(1 - sigma) * 1000`. Same table,
   different argument; borrowing the wrong one is finite nonsense.

Also: `concat_padding_mask` appends an all-zero **17th channel** before patchifying
(which is why `x_embedder.proj.1.weight` is `[2048, 68]`); one AdaLN-LoRA vector is
shared by all three sublayers and the final layer takes only its first `2*dim`;
`adaln_modulation_*` is `Sequential(SiLU, Linear(d, 256), Linear(256, 3d))` with **no
activation between the two linears**; RoPE applies to DiT self-attention only but to
**both q and k** in the adapter's cross-attention; and the RoPE frequency vector is
`[t(22) | h(21) | w(21)]` with h/w on an NTK-scaled theta `10000 * 4^(42/40)`.

### The T5 tokenizer is a real piece of SentencePiece

`core/t5_tokenizer.zig` — Unigram + Viterbi, with the `Precompiled` charsmap
normalizer (SentencePiece's `nmt_nfkc`: a darts-clone double-array trie, embedded
verbatim at 237 KB) plus `Strip(right)` and `Replace(/ {2,}/ -> U+2581)`, then
`Metaspace`. It is not an approximation: the charsmap maps NBSP/ZWSP/**ZWJ**/
ideographic space to a plain space, deletes 30 control characters, folds ligatures
(`ﬁ`→`fi`), fullwidth Latin, superscripts and circled digits, expands `Ⅸ`→`IX` (one
char to **two**), and composes `e`+U+0301 into `é`.

⚠️ **It is NOT a variant of `tokenizer.zig`'s existing `.unigram` kind** (added for
Snowflake Arctic Embed). That one splits on ASCII whitespace and prepends `▁` to
*every* word, has **no normalizer at all**, and its lattice has no `<unk>` nodes — an
unsegmentable word becomes one `<unk>` for the whole word where `tokenizers` inserts
one per unsegmentable *character* at `min_score - 10` and then fuses runs. Its
tie-break also prefers the shortest incoming piece where `tokenizers` prefers the
longest. **Those last two look like latent bugs in the embedding tokenizer** — its own
fixture passes, so its corpus evidently never exercises a partially-unknown word — but
they were deliberately NOT changed, since altering them would move every embedding
this repo has computed.

⚠️ **Five ComfyUI conventions, each a silent wrong answer**, all pinned by
`tools/gen_anima_prompt_fixtures.py` (50 cases, generated by *executing*
`AnimaTokenizer`):

- **Weighted segments are tokenized SEPARATELY** — one tokenizer call each. ⚠️ A split
  at a SPACE is transparent (the segment's own `add_prefix_space` reproduces the `▁`
  the space would have become), so **only a mid-word split reveals a wrong port**:
  `cat(s:1.1)` is `▁cat ▁s`, never `▁cats`. The fixture's own teeth-check asserts at
  least one case distinguishes them, and it caught the first version, where every case
  happened to split on a space.
- **`</s>` appears once, at the end**, while each *segment's* is dropped
  (`input_ids[0:-1]`). `end_token` is 1 because `SDTokenizer.__init__` resolved it from
  `tokenizer("")["input_ids"][0]`, not because anything configured it.
- **The `▁` prefix is added iff the span starts at byte 0 of the call's input** —
  Metaspace's `prepend_scheme = "first"` tests `offsets_original().0 == 0`, NOT "is
  this the first split". So `prefix<extra_id_0>suffix` ends `… [s][uff][ix]`, with no
  `▁`. Breaking exactly this rule is what the fixture catches first.
- **An empty normalized span produces NOTHING**, not a bare `▁`. `" "`, `"   "` and
  `"\x01"` all tokenize to just the trailing `</s>`.
- **Unknown pieces FUSE** within a pre-token but not across one: the two Gothic letters
  of `𐌰𐌱` are one `<unk>`; separated by a space they are two.

⚠️ **The Qwen3 branch is ALSO per-segment**, and finding that out cost the first
end-to-end comparison. `tokenizer.encodeSegmented` exists for it. Measured: the real
positive prompt's whole-string and per-segment encodes diverge from **token 37**,
because a segment ending in `", "` leaves the trailing space as its own token 220 where
the whole-string encode merges it into the next word.

`core/prompt_weights.zig` was extracted from `clip_tokenizer.zig` so the `(a:1.2)`
parser has ONE implementation — two copies of "a bare paren multiplies by 1.1, an
explicit `:w` replaces" is exactly the drift that makes one prompt path disagree with
another.

### Measured

Component parity against ComfyUI, fp32 on both sides (`tools/gen_anima_fixtures.py`;
the trunk truncated to 8 and 2 blocks so an *exact fp32* reference on real weights at
real width fits in RAM, the adapter referenced in **full** because at 133 M parameters
there was no reason not to). The table is in `models/anima.zig`'s header;
`patchify + x_embedder` is **bit-identical**, everything else 1e-8…8e-6, and ⚠️ **the
DiT's disagreement is FLAT in depth** — 2.2e-6 at 2 blocks, 1.9e-6 at 8 — so nothing is
accumulating, which is what says the block form is right rather than merely close.
(Z-Image's grows linearly and that was already good enough.)

**End to end**, against ComfyUI **fp32** at 256²/4 steps/seed 42, same prompts, same
sigmas: **78.44 dB / SSIM 1.0000 at cfg 1.0** and **67.31 dB / SSIM 0.9999 at cfg 4.0**,
max difference **1/255** either way. That is the f32 reduction-order floor — there is no
residual to explain.

⚠️ **The 15 dB "disagreement" that was entirely the harness, and it is the same trap
this file already records for Z-Image.** The first comparison read 15.3 dB with our
render visibly washed out. Cause: `tools/render_anima_ref.py` applied Wan21's
`process_out` (`z * latents_std + latents_mean`) itself, but
`CFGGuider.sample` already ends with `self.inner_model.process_latent_out(samples)`
(`comfy/samplers.py`) — so the reference denormalized **twice**. The composition
survived intact — same subject, same pose, same colours in the same places — and only
the tone shifted, which reads as a numerics problem in the engine. **The tell is the
signature: structure matching while tone does not means the trajectory agreed and only
the output mapping did not.** The Z-Image script's own comment says exactly this and I
overrode it.

**End to end at the REFERENCE RENDER's own settings** — 512x768, 30 steps, cfg 5.0,
euler + `normal`, seed 80085, its 125-T5-token positive prompt (parenthesised emphasis
included) and its negative. ⚠️ **Read the control row first:**

| | PSNR | SSIM |
|---|---|---|
| **CONTROL** — ComfyUI bf16 vs ComfyUI fp32 | **25.16 dB** | 0.9448 |
| **TensorPencil f32 vs ComfyUI fp32** | **44.39 dB** | **0.9990** |
| TensorPencil f32 vs ComfyUI bf16 | 25.19 dB | 0.9451 |

44.39 dB is **19 dB inside the reference's own precision envelope**, and the last row is
statistically identical to the control — i.e. against a bf16 render this engine is
indistinguishable from a second ComfyUI fp32 render.

⚠️ **This model is PRECISION-DOMINATED at cfg 5, so read the control first.** At
1056x1584 / 30 steps / cfg 5 / euler + `normal` / seed 80085, **ComfyUI disagrees with
itself**:

| | PSNR | SSIM |
|---|---|---|
| ComfyUI bf16 vs ComfyUI fp32 (one harness) | **23.04 dB** | 0.9020 |
| the user's own bf16 render vs my bf16 harness | 25.04 dB | 0.9269 |
| the user's own bf16 render vs my fp32 harness | 23.52 dB | 0.8925 |

23–25 dB is therefore the **ceiling** for any comparison against that PNG, and it sits
right on Z-Image's documented 24.49 dB bf16-vs-fp16 floor. (The two bf16 renders differ
because the reference harness forces `--use-pytorch-cross-attention` where a normal
ComfyUI server picks its own attention kernel.)

### On the GPU

**Every backend runs the whole pipeline** as of 2026-08-07 (`anima_gpu.zig` /
`anima_cuda.zig`). At 512x768 / 30 steps / cfg 5 on a 3090, **`cuda` is 0.37 s/step
against ComfyUI's 0.33 bf16** — 89% of it, and **118x** the CPU arm's 43.60. At
1056x1584 it is 1.62 against ComfyUI's 1.31 (81%). BACKEND.md 2C has the full grid,
the per-component bench and the parity table; what belongs here is what the port cost.

| | 512x768 | 1056x1584 | vs ComfyUI fp32 (control 25.16 dB) |
|---|---|---|---|
| `cuda` | **0.37 s/step** | **1.62 s/step** | 33.36 dB |
| `zig-cuda` | 0.42 | 2.32 | 35.32 dB |
| `vulkan` | 0.59 | **3.11** | 28.95 dB |
| `cpu` (f32) | 43.60 | 3-8 min | 44.39 dB |

⚠️ **Read the control first.** All three device arms sit inside ComfyUI's own
**25.16 dB** bf16-vs-fp32 envelope at these settings, and they agree with each other at
28-36 dB — the same band, i.e. what separates them is precision, not a defect in one. At
1056x1584 (control **23.04 dB**) they are **32.29 / 33.84 / 33.71 dB**, and `vulkan` —
the lowest of the three above — is the *highest* there, which is what settles that the
spread is chaotic rather than systematic.
Unit-level, all three match `anima.DiT.predict` at **8.0e-4** rel L2 (Vulkan 8.02e-4,
zig-cuda 8.10e-4, cuda 8.05e-4) — two independent implementations landing on the same
figure, and **the attention choice is not what the residual is made of**: forcing the
naive kernels moves it to 8.07e-4.

**Two new kernels and one op generalization.** Everything else was already there.

- ⚠️ **`ln_mod_sg` is a SUBGROUP kernel, and that is the point.** Fused weightless
  LayerNorm + AdaLN modulation, one subgroup per row. A thread-per-row form at 6534 rows
  x 2048 wide x 3 calls x 28 blocks is precisely the trap `qk_rmsnorm_warp` already paid
  for (37 GB/s on a 936 GB/s card, a quarter of a Z-Image step). It computes the variance
  in **two deviation-based passes**, matching `ops.norm.layerNormUnit` rather than the
  shifted `E[x^2]-E[x]^2` form — `ops.norm.groupNorm` records why, and the device test
  includes a row whose mean is 400 against a spread of 1.5 for exactly that reason.
  ⚠️ At that mean the **f32 host reference is the less accurate side** (its serial sum of
  2048 values near 400 carries ~5e-4), so both are scored against f64 and the device is
  required to be no worse — a comparison against `layerNormUnit` directly *fails the
  correct implementation*.
- **`ln_mod_par`** is its CUDA twin, **derived from `ln_bias_par` by asserted
  substitution** (`replaceOnce`, which `@compileError`s on an absent or non-unique
  pattern) rather than copied, so the block reduction and the two-pass variance stay
  literally shared.
- ⚠️ **`coopmat.buildFlashAttn` gained an MD plane stride (push word 7), which is what
  makes it RECTANGULAR.** `s_stride` was K's row stride, the j loop bound *and* the MD
  plane stride at once; the MD table is indexed by *query* row, so once `q_pad` exceeded
  `kv_pad` every head past the first read another head's `{max, 1/sum}`. **0 means
  "= s_stride"**, so every pre-existing caller is unchanged by construction — and the
  teeth check is exact: forcing f1 = 0 fails at row 128, the first row past `kv_pad`.
  `buildGemmAttnOut` already had a `pmdplane`; this is its sibling catching up.
- **`opAttnTCRect`** is `opAttnTCBatched` with the two sequence lengths pulled apart. The
  launches were already parameterized by m/n/k with independent per-head strides, and
  `launchAttnOut` already documented that its `k` "== m only for square attention".

⚠️ **Rectangular tensor-core attention is a REQUIREMENT, not an optimization.**
Cross-attention is `seq x 512` — **2.4% of a step's FLOPs**, 5.6% of its measured time —
so the temptation is to leave it on the naive thread-per-(query, head) kernels. But each
of those threads streams the whole 512-key context, which estimates to ~0.6 s/step at
1056x1584, i.e. **a third of the whole step for 2.4% of the work**. The FLOP share is not
what decides; the kernel's data reuse is.

**Cross-attention's K and V are per-IMAGE constants, precomputed for every block at
`Session.init`.** They are projections of the adapter's output, which no step changes, so
28 blocks x 2 GEMMs of `[512, 2048] x [2048, 1024]` leave the step loop. Vulkan caches
them directly in the **f16 attention-operand layout** (117 MB), which also removes the
per-step conversion and the k-major gather; CUDA's entry points convert internally, so its
cache is f32 (235 MB). The K norm is per-image too and is applied before caching. Same
observation `zimage_gpu` makes about its `modulation=False` caption half. `DiT.crossKv` /
`projectKv` exist so "what K and V are" has ONE implementation — two copies of "project,
then norm K but not V" is exactly the drift no shape check would see.

⚠️ **A 30x arithmetic error is recorded here rather than quietly fixed, because the shape
of the mistake is the lesson.** This optimization was motivated by "~17% of the trunk's
per-step GEMM FLOPs", which came from dividing a **28-block total** by a **per-block**
figure. The true share is **0.56%** at 6534 tokens (4.29 GFLOP against a block's 767) and
2.4% at 1536 — obvious in hindsight, since these two GEMMs run over the 512-row *context*
where every other GEMM in the block runs over the *sequence*, and 512 is a twelfth of
6534. **Measured** by `anima-cuda-bench`, which times the hoisted GEMMs explicitly and
outside the step roll-up: 0.10 ms/block, i.e. the hoist is worth **0.38% of a step**. So it
is a small win that costs 117-235 MB (doubled under CFG), kept because it also removes a
per-step conversion, and **not** what makes this port fast. The general rule this violated:
**a ratio between two numbers computed at different scopes is not a ratio.** Per this
file's own standard the receipt should have come before the code, not after it.

⚠️ **The cost of that cache is that a session is bound to ONE conditioning, so CFG needs
two of them** — 235 MB of cross K/V on CUDA and 117 MB on Vulkan becomes twice that at
cfg > 1. Measured VRAM at 1056x1584 / cfg 5: `te=840MB dit=3527MB latent=1051MB`, ~5.4 GB
resident, so it is comfortable on a 24 GB card and would be the first thing to reconsider
on a small one. It is the same shape of trade Z-Image's per-conditioning sessions already
make.

**Three bugs the port cost, and two are recorded elsewhere in this file in other forms:**

- ⚠️ **The folded modulation table must NOT fold the FINAL layer's scale.** The device's
  fused norm has no place for the `1 +`, so every *block* scale arrives pre-folded — but
  both arms run the final layer on the HOST through `DiT.finalize`, whose `modulatedNorm`
  adds its own 1. Folding it there too gave **rel L2 0.10** against the CPU forward:
  finite, plausible in magnitude, and **identical on both attention paths**, which is
  what said "shared wiring, not attention" and found it in one step. The rule is *fold
  exactly what the fused device norm consumes*.
- ⚠️ **`offsetBuf` is a CUDA idiom and does not port.** A Vulkan `DeviceBuffer.buf` is an
  opaque HANDLE, not a device pointer, so `zimage_cuda`'s trick of adding a byte offset to
  it produced an invalid handle and `error_device_lost` on the first cross-attention. The
  tell was the *size*: a 24x32 latent is far too small for the watchdog to be plausible,
  so a device loss there is a fault, not a timeout. Fixed with **one buffer per block**,
  since a descriptor binds a whole buffer.
- **`replaceOnce` needs `@setEvalBranchQuota`** for a PTX-sized haystack — `std.mem.indexOf`
  runs Boyer-Moore at comptime.

**Where the step goes, and why the answer is "nowhere unexpected"**
(`anima-cuda-bench 6534 libs`, one forward): GEMM ~440 ms at 49 TFLOP/s effective and
**49-54 pure**, self-attention 190-220 ms at 44 TFLOP/s, cross-attention 44 ms, elementwise
78 ms at **410-830 GB/s** — total **750-790 ms** across runs. ⚠️ **The isolation that makes that a receipt
is CFG**: 787 ms is ONE forward, a cfg-5 render does two, and the same size at `--cfg 1.0`
measures **0.82 s/step** against the predicted 0.787. So 2 x 0.82 = 1.64 against the 1.62
measured — nothing is hiding, and every component is healthy against its own ceiling.
Closing the remaining ~19% would mean **bf16 activations end to end** plus better GEMM
tiling at these widths; named, not attempted.

⚠️ **The device arms serve conditioning entry 0 only.** A device session caches
cross-attention's K/V for the conditioning it was built with, so an A1111 per-step prompt
schedule (`[a:b:0.5]`, `[a|b]`) would need one session per variant. `predictAnima` detects
a scheduled entry and falls back to the host forward rather than silently rendering the
wrong prompt — the same reason krea2's and Z-Image's device paths read entry 0, made
explicit here because Anima *does* support scheduling.

### int8/int4 convrot: the CPU runs it, the GPU arms refuse it BY BLOCK

`anima.zig`'s loader wires the convrot metadata (`<name>_weight_scale` per output row plus
the 256-wide rotation), so an int8/int4 Anima checkpoint runs on the **CPU** — `ops.matmul`
already had the dequant-and-rotate paths. The `comfy_quant` blob confirms the convention
verbatim: `{"convrot": true, "convrot_groupsize": 256, "per_row": true, "format":
"int8_tensorwise"}`.

⚠️ **Omitting that wiring is a PANIC, not a slow path.** `ops.matmul.matmul` asserts
`row_scale != null` for an integer weight, and the loader accepted the I8 tensor because
`supportsDType(.i8)` is true — so it fired five frames deep, inside `crossKv`.

⚠️ **`unsupportedGpuLin` scans EVERY block's linears, because checking one tensor of one
block is wrong on a real checkpoint.** `anima_baseV10-INT8_CONVROT-MIXED` keeps **block 0
entirely bf16** and quantizes blocks 1-27 (block 1: the 10 attention/MLP linears only;
blocks 2-27: all 16, AdaLN included). A `supported()` that read
`blocks[0].self_attn.q.dtype` therefore said "GPU ok", built a device session, and crashed
on the first thing it did. **"Mixed" means mixed per block**, and per-block-uniform is the
easy case rather than the general one. `zimage_cuda.supported` still reads one tensor —
correct only because no mixed Z-Image checkpoint exists yet.

The warning names the offending layer (`block 1's blocks.1.self_attn.q_proj.weight is
i8 …`), because "unsupported dtype" is unactionable when the answer to "but my checkpoint
is bf16" is "block 1 is not".

**int8 and int4 convrot run on the DEVICE on all three GPU backends** as of 2026-08-07,
with the kind resolved **per block**. Measured at 512x768 / 20 steps / cfg 4 on a 3090
(`easonAnimaHOTStyle_animaV10`, its own bf16 as the reference):

| | `cuda` s/step | DiT VRAM | vs its bf16 render | `vulkan` s/step | DiT VRAM |
|---|---|---|---|---|---|
| bf16 | 0.43 | 3230 MB | — | 0.57 | 3272 MB |
| **int8** | **0.23** (1.9x) | **1781 MB** | **25.8 dB / SSIM 0.936** | **0.48** (1.2x) | **1779 MB** |
| **int4** | **0.18** (2.4x) | **1025 MB** | 11.9 dB / SSIM 0.451 | **0.48** | **1779 MB** (= int8's) |

int8 is free in the way that matters — same composition, same colours, same detail, at half
the weights and nearly double the speed on `cuda`. int4 keeps the *composition* exactly and
destroys the surface, which is what the user predicted and what quantization damage looks
like; a broken kernel gives noise or a flat field, not the same pose with a grainy skin.

⚠️ **`dit_cuda`'s `LinKind` is one value for the whole model ("uniform across blocks") and
that is WRONG here.** A real mixed checkpoint is mixed by block:
`easonAnimaHOTStyle_animaV10-INT8_CONVROT` leaves block 0 entirely dense, quantizes block
1's ten attention/MLP linears, and quantizes all sixteen (AdaLN included) in blocks 2-27. So
`anima.linKind` is per linear and `prepGroup` is per shared activation. The AdaLN pair is
quantized too but is evaluated on the HOST, where `ops.matmul` handles convrot at any shape —
which is what lets the device path require `rows % 128 == 0` (CUDA) / `% 64` (Vulkan) without
special-casing its 256 and 6144.

⚠️ **Cross-attention's k/v are on the DEVICE and were missing from that scan until
2026-08-08.** They *were* host GEMMs when `unsupportedLin` was written, and the comment saying
so outlived the change that moved them (the per-image stall fix below) — so a checkpoint whose
cross k/v were quantized in a form the backend lacks would have passed the gate and met it
inside `buildCrossKv`. That is the block-0-only probe's mistake exactly, one level down, in the
very function that exists to prevent it. `anima.deviceLins` is now the single list all three
scans (`unsupportedLin`, `maxNvfp4Scratch`, `maxW4A8Scratch`) read. **Generalizable: when a
computation moves from host to device, the SUPPORT SCAN is part of what moves.**

⚠️ **`opI8Prep` does NOT overwrite its input** on either backend — it writes int8 rows and
per-row scales into the backend's own scratch — so a dense GEMM in the same group may still
read the f32 activation afterwards. The only constraints are that the prep precedes the
quantized GEMMs of its group and that one prep serves one reduction width. Anima's MLP is
8192 wide where everything else is 2048, so `mlp2` gets its own prep.

⚠️ **Vulkan has NO `sint4` cooperative matrix**, only `sint8`, so the nibbles must reach the
GEMM as int8 either way and the activation stays int8 (`opI8Prep`) — i.e. Vulkan's int4 is
W4A8-shaped, not the CUDA arms' W4A4, and is in fact *more* accurate for it.

**Until 2026-08-08 that unpack happened ONCE AT LOAD** (`anima_gpu.Lins.widen`), which keeps a
full int8 copy resident — so a 4-bit checkpoint cost 8-bit memory. It now happens **per GEMM**
into `ctx.i4_t` (`i4_decode_t` / `Context.i4Decode`), and `Lins`/`DevLins`/`widen` are gone
entirely: `blockForward` reads the model's own weights again.

| `terraRising`, vulkan, 512x768 / 20 steps / cfg 4 | DiT VRAM | s/step |
|---|---|---|
| int4, widened at load | 1778 MB | 0.48 |
| **int4, decoded per GEMM** | **1038 MB** | **0.46** |
| (W4A8, for scale) | 1118 MB | 0.49 |
| (int8) | 1778 MB | 0.47 |

**-740 MB and bit-identical** — the render md5 is unchanged, as it must be, since a 4-bit value
is exact in 8 bits and the scale and rotation are untouched. int4 now sits *below* W4A8, which
is the right ordering: it carries 2.9 MB of f32 per-row scales where W4A8 carries 109 MB of fp8
per-group ones.

⚠️ **What made this worth doing was noticing the two formats were treated differently for no
reason**, and the arithmetic is what exposed it: the device linears hold **1050.7 MB of packed
weight in BOTH formats** (both are nibble-packed 4-bit), so int4 held strictly *less* data than
W4A8 and yet occupied 660 MB more. A footprint difference with no corresponding data difference
is a policy, not a property — worth checking whenever two similar formats disagree on VRAM.

⚠️ **The old cost estimate here was badly wrong and is worth remembering as a pattern.** It read
"a nibble-reading coop GEMM or an i4→i8 repack … both are real kernel work, neither done" — true
when written, and never revisited after `w4a8_decode_t` landed and made it a ~10-line
simplification (same pre-transposed `weightBuffer` input, same `[cols][stride]` int8 output,
same `opI8GemmBuf` after; just no scale plane and no level table). **An effort estimate has an
expiry date set by what else gets built**, and a "not done, too expensive" note is exactly the
kind that stops being re-examined.

**Pinned by `test "the Vulkan int4 decode matches the CPU nibble unpack, including the row
padding"`** — the reference is precisely the unpack `widen` used to do, so the test asserts the
replacement rather than merely something plausible. Verified to have teeth by swapping the
nibble halves. ⚠️ It checks the row PADDING too: the scratch is shared between weights of
different shapes, so an unwritten pad slot would feed the next GEMM the previous weight's values.

**krea2's Vulkan DiT still has no int4 path at all** (`dit_gpu` never accepted it), and this
kernel is now most of what it would need. Not attempted.

⚠️ **That widening must be per MODEL, and making it per session doubled the VRAM.** The
device weight cache keys on the host POINTER, so one shadow per `anima_gpu.Session` uploaded
a second full copy of every quantized weight for the CFG-negative branch — measured
**`dit=3291MB` for int4 against int8's 1779 MB**, i.e. the widening looked twice as
expensive as it is, and the unpack ran twice per image. `pipeline.Session` owns one `Lins`.

⚠️ **Anima's reduction widths had to be added to `gpu.i8_prep_cols`.** The Vulkan fused prep
kernel unrolls its FWHT for a fixed `cols` and only krea2's 6144/16384 existed; a missing
width silently takes a 3-pass fallback that round-trips a full f32 copy of the activation
through global memory. The table is now `{2048, 6144, 8192, 16384}` and looked up rather
than switched on.

### W4A8 on Anima: a format that shipped for one family and arrived on another

Landed 2026-08-08 for `terraRising_20TerraRisingAnima-ASYM_W4A8_INT8.safetensors`, reported
as "this one has an error when running it in the gui (maybe CLI too?)". It was both — the
failure is at *load*, so every surface hit it:

```
error: anima: blocks.1.self_attn.q_proj.weight is i8 but
       blocks.1.self_attn.q_proj.weight_scale is missing
       (int8/int4 convrot needs a per-row scale)
```

⚠️ **The bug is that W4A8 landed in `dit.zig` alone, one day earlier, and the format is not
krea2's.** A W4A8 weight is stored `I8 [rows, cols/2]`, which is *bit-for-bit* the int4-convrot
signature — so Anima's loader, which had never heard of the format, fell through to its int4
arm and died on the `_scale` W4A8 does not have (its spelling is `weight_s_channel`). **That
was the lucky outcome.** Had the names matched, the nibbles — unsigned indices into a
non-uniform Lloyd-Max codebook — would have been read as signed int4 times a per-row scale:
finite, plausible and wrong. `dit.zig`'s own comment predicted exactly this ("it would in fact
fail on the missing `_scale` today, but that is an accident of naming, not a check") and the
conclusion drawn from it was to write the comment rather than to move the reader.

**The fix is where, not what.** `models/quant_weight.zig` already existed for precisely this
reason — NVFP4 arrived on three architectures at once and got one shared container reader —
and W4A8 simply was not put there. It is now (`quant_weight.w4a8`, hoisted verbatim out of
`dit.Loader.matW4A8`), called from krea2's, Anima's **and Z-Image's** `mat`. ⚠️ **Z-Image is
included although no such checkpoint exists for it**: nothing about that family would have
stopped one arriving, and `.w4a8` is absent from its `gpuLinKindSupported`, so it would run on
the CPU while every GPU arm declines — a strictly better state than a hard error.

⚠️ **Generalizable, and this is the whole content of the section: every format ComfyUI's
quantizers emit reaches every family they support.** A container reader for one of them
belongs in the shared module the day it is written, not after the second checkpoint arrives.
The cost of getting it wrong is not a slow path; it is a file the engine refuses.

**Verified bit-identical across the hoist**: krea2's own W4A8 checkpoint renders the same md5
before and after (512², 4 steps, seed 42, `cuda`), A/B'd by temporarily restoring the old
inline path. ⚠️ The obvious control — `test "a W4A8 layer of a real checkpoint decodes to
comfy_kitchen's int8 weight"` — could **not** serve, because the file it is tied to by sha256
is gone from this box and it had been silently self-skipping. The file with a similar name
(`v10Int8AndBf16-` against the fixture's `v10Int8-`) has *matching shapes and different bytes*,
so repointing it would have compared against the wrong weights while passing green. Same trap
this file already records for the Anima gated tests.

**The compute side needed no new kernel on any backend** — W4A8's activation prep *is* int8's
and its decode output *is* an int8-convrot weight, so all three GPU arms already had
`opI8GemmW4A8` / `w4a8Decode` from the krea2 work. What Anima needed was the per-block
dispatch: `LinKind.w4a8`, the scratch pre-sizing, and one new concept —

⚠️ **`anima.prepKind`, because `LinKind` is NOT one-to-one with the activation prep.** int8 and
W4A8 share `opI8Prep`; grouping by `LinKind` would have refused this very checkpoint as an
unserviceable mix (its block 1 has W4A8 attention weights next to block 0's dense ones). The
prep is a property of the *activation*, the kind a property of the *weight*, and only the
second is what the GEMM dispatches on.

| `terraRising`, 512x768 / 20 steps / cfg 4, `cuda` | s/step | DiT VRAM | vs its own bf16 |
|---|---|---|---|
| bf16 | 0.44 | 3230 MB | — |
| int8 convrot | 0.28 | 1781 MB | 32.31 dB / SSIM 0.978 |
| **w4a8** | **0.26** | **1120 MB** | **20.86 dB / SSIM 0.856** |

**So W4A8 runs at int8's speed on 63% of its VRAM**, the same shape as krea2's result. The
three GPU arms agree with each other at **34.9–38.2 dB / SSIM 0.987–0.996**, which is what
says the spread against bf16 is format loss and not any one arm's kernel.

⚠️ **The parity harness FAILED a correct implementation, and the fix was the tolerance — but
only after an isolation said so.** `anima-cuda-test` read **2.0968e-3** against a 1e-3 bound,
because `.w4a8` fell into the *dense* arm of the tolerance switch. W4A8 decodes to int8 and
then runs int8's prep and GEMM, so it must land where int8 lands — and the same model
quantized int8 measures **2.0721e-3**, a 1.2% gap, with both attention arms agreeing to three
digits and depth-1 (dense block 0) bit-identical across the two files. Vulkan independently
reads **2.1004e-3**. **Widening a bound to make a test pass is only legitimate when the
predicted value is known first**; here it was, and the receipt is a sibling checkpoint rather
than an argument.

**Pinned by two new tests, one of each tier.** ⚠️ The ungated one is the important one — a
synthetic store fed through `anima.Loader.mat`, asserting the weight loads as `.w4a8` and not
`.i4`, that the packed bytes and `s_rel` stay VIEWS (the whole point of the format), and that
the decode produces the right int8 for a deliberately non-uniform codebook whose level 1 is
`-3.5`, so it pins ties-to-even too. **Verified to have teeth: with the loader call removed it
fails with the user's exact `MissingTensor`.** The gated one adds the real W4A8 checkpoint to
the Vulkan forward-parity test, and was likewise confirmed to run rather than self-skip.

### `opI4Prep` wrote NOTHING for a 1024-wide reduction

Fixed 2026-08-08, reported as "it renders just fine in comfyui, something must be wrong with
the tp code" against a ComfyUI render of the same seed. It was, and the first diagnosis in
this file was wrong — see the postmortem at the end of this section.

**The bug is one integer division.** `kernels.buildPrep` sized the packing loop as

```zig
const word_iters = cols / (per_word * 256);   // per_word = 32/bits: 4 for s8, 8 for s4
```

At `bits == 4` a 1024-wide row is **128 packed words for 256 threads**, so `word_iters` is
**0**, the store loop is emitted zero times, `p_q` is never written, and `opI4Gemm` multiplies
the weight against whatever the shared int8/int4 activation scratch held from the previous
call. No error, no assert, nothing non-finite.

⚠️ **Only one linear in the repo is that narrow: Anima's cross-attention k/v (`context_dim`
= 1024).** Every krea2 width (6144, 16384) and every other Anima width (2048, 8192) divides
evenly, and int8 (`per_word` = 4) is safe at 1024 too — which is why this survived: it is
reachable by exactly one dtype at exactly one width, on one architecture. It became reachable
at all only when cross k/v **moved from the host to the device** (the per-image stall fix
below), so the code that broke it is not the code that contains it.

The fix is `word_iters = ceil(total_words / 256)` plus a per-thread `word < total_words`
guard, emitted **only** when `total_words % 256 != 0` — i.e. only for this one case. Every
previously-working width keeps the same iteration count and the same instructions.
`nbf == 0` (cols < 1024, where the FWHT itself would be skipped) is now `error.UnsupportedWidth`
rather than a silently non-rotating kernel; no live linear is that narrow.

| | before | after |
|---|---|---|
| int4 `cuda` vs its own CPU forward, 256² render | 5.95 dB | **24.46 dB** |
| int4 `zig-cuda`, same | 5.05 dB | **23.20 dB** |
| int4 `vulkan` (never took the broken path) | 30.03 dB | 30.03 dB (unchanged) |
| TP int4 vs **ComfyUI's own int4**, its 1056x1584/30-step workflow | 4.48 dB | **14.63 dB** |
| krea2 int4 render | — | **byte-identical** (A/B'd across the change) |

⚠️ **Vulkan was the control that localized it, and it is a control by accident.**
`anima_gpu.Lins` widens int4 to int8 because there is no `sint4` coopmat, so it runs
`opI8Prep` and never touches the broken path — CPU and Vulkan clean, both CUDA arms static.
That immediately ruled out the weight decode, the nibble order, `weight_scale` and the convrot
basis (Vulkan exercises all four), leaving only the int4 **activation** path. Worth
remembering: a backend that implements a format *differently* is a free bisector.

### ⚠️ The postmortem: why the first diagnosis said "found, not caused"

This file previously claimed int4's breakage was "depth accumulation of W4A4 specific to this
quantization", explicitly not a TP defect. That was wrong, and every step of the reasoning is
worth keeping because each looked sound:

1. **The depth-2 parity check passed at 3.86e-2 — int4's own documented coarseness figure.**
   The harness only ran depths 1 and 2, justified in a comment reading "depth adds no new
   WIRING". True of the wiring; false of the *defect*, which corrupts only cross-attention
   and so needs depth to show. ⚠️ **A single depth cannot distinguish "coarse" from
   "accumulating"**, and that distinction was the entire question. `anima-cuda-test` now
   sweeps `{1, 2, 8, 28}` with a depth-scaled bound; healthy int8/W4A8/int4 are flat from 2
   to 8 (2.1e-3, 2.1e-3, 3.9e-2) where the broken kernel climbed to 1.97e-1 at depth 8 and
   5.2e-1 at 12. Verified to have teeth by reintroducing the bug: 4 failures.
2. **The CPU-vs-GPU comparison that "exonerated" the kernel was CONFOUNDED** — CPU at 256²
   against CUDA at 512x768, two variables at once. Re-run at one resolution it says the
   opposite immediately.
3. **The reference was never consulted.** ComfyUI renders this checkpoint's int4 badly too
   (9.47 dB / SSIM 0.252 against its own bf16), and that superficial agreement is what made
   "the checkpoint is just bad" feel confirmed. But ComfyUI's is degraded-**with-structure**
   and TP's was structureless static — a qualitative difference visible in two seconds of
   looking, which no aggregate I computed would have surfaced. **"Both are bad" is not
   "both are equally bad."**

**Generalizable:** a negative conclusion needs an isolation that removes exactly the component
in question, and "the device matches the CPU at depth 2" does not isolate a defect that only
manifests at depth 28. When a user says the reference does it better, render the reference's
exact workflow and *look at both images* before reasoning about the numbers.

⚠️ **The fix also moved the headline quality figure, and by more than the bug's "one
narrow linear" framing suggests.** At 512x768 / 20 steps / cfg 4 against this model's own
bf16, int4 went **2.32 dB / SSIM 0.191 -> 20.45 dB / SSIM 0.837** — i.e. from unusable to
essentially level with W4A8's 20.86 / 0.856 at a further 149 MB saved. Two GEMMs per block
out of sixteen were corrupt, and that was the whole difference between "this format is
broken" and "this format is competitive". ⚠️ **Worth remembering when triaging a quantization
arm: a small number of wrong linears does not produce proportionally small damage**, because
the corruption re-enters the residual stream every block and then gets amplified 30 times by
the sampler.

int4 is still the coarsest arm and still degrades this checkpoint more than int8 does
(32.31 dB), but it is no longer in a different category — and ComfyUI's own int4 of the same
file is *worse* than TP's is now.

### "Anima images are slow to start": three candidates, and it was the third

Reported as "it looks like it is loading the text encoder on the GPU, then doing the encoding
on the CPU". The encoder was on the GPU. An `[encode]` log line now states where each half
ran, because the answer was not guessable and I guessed wrong twice.

Measured at 512x768 / 30 steps / cfg 5 on `cuda`, before any fix:

| | | |
|---|---|---|
| qwen3 encode on `cuda` | 1.15 s first call, **0.03 s second** | one-time weight upload + PTX JIT, not host work |
| `llm_adapter` on the host | **0.17 s** | cheap, as designed — not the problem |
| **per-image device session build** | **1.24 s** | ← the actual cost |

⚠️ **Both halves of that 1.24 s were my own per-step reasoning applied to a per-image cost.**

1. **Cross-attention K/V were projected on the HOST** — 28 blocks x 2 GEMMs of
   `[512, 2048] x [2048, 1024]`, per conditioning. The old comment justified it with "one
   512-row pass per block, done once per image", and as a share of a *step* it is 0.56%. As a
   one-off before the first step it was hundreds of milliseconds, every image. Now device
   GEMMs (`prepGroup`/`lin`, so a quantized checkpoint works too), which also means the cross
   `k`/`v` weights get uploaded (~235 MB) where before only the host read them — a trade the
   original comment named and I had priced only in per-step terms.
2. ⚠️ **The AdaLN modulation schedule was built PER CONDITIONING, and it depends on the sigma
   alone** — so the positive and negative sessions computed byte-identical tables:
   **0.51 s + 0.29 s** at 31 sigmas. `anima.DiT.modulationSchedule` builds one and
   `pipeline.Denoiser` owns it; both sessions borrow.

**Result: setup before the first step 2.9 s -> 0.7 s** (session build **1.24 -> 0.32 s**),
sampling unchanged at 0.35 s/step, quality slightly *better* — **33.72 dB** against ComfyUI
fp32 where it was 33.36, because the k/v projections now run in the same tensor-core regime
as everything else rather than in host f32.

**Vulkan's cross K/V moved to the device too** (`gather_kmajor_h16` / `f32_to_h16` straight
into the per-block f16 cache; `DevLins` gained `ck`/`cv` so an int4 checkpoint widens them like
the rest). Forward parity unchanged, 14/14 gated.

⚠️ **But it did NOT reduce Vulkan's single-image setup, and the measurement says why:
0.69 s for the first conditioning against 0.05 s for the second** — a 14x gap for identical
work, so that 0.69 s is **one-time** (the ~235 MB cross-k/v weight upload plus first-use shape
warm-up), not per-conditioning. Net effect: a single image is roughly unchanged (0.71 ->
0.74 s), while every image after the first drops from ~0.6 s of host GEMM to **~0.10 s**,
because `ctx.weights` outlives the per-image session. So it is a queue win, not a first-render
win, and saying "moved to the device, therefore faster" would have been wrong.

⚠️ **What that exposes is a Vulkan WEIGHT-UPLOAD gap, not a cross-K/V one.** CUDA absorbs the
same 235 MB inside a 0.32 s total build; Vulkan spends 0.69 s on it. CUDA's path is pinned
staging fanned over `Context.FillPool`'s 4 threads (the fix that took Z-Image's step 1 from
5.1 s to 2.6 s); Vulkan's has no equivalent. That is the next thing to measure — and it would
help every model, not just Anima.

⚠️ **The legacy krea2 `encodePrompt` free function is gone**, folded into `runQwen3`. It
called `qwen3_cuda.encode` **unguarded** — that arm reads its weights as fp8, so a bf16 krea2
encoder would have become noise rather than being refused — and it hardcoded
`qwen3.tap_count * qwen3.hidden`, the 4B module constants that the width generalization made
wrong for any other body. There is now one Qwen3 dispatch for all three families.

**The generalizable rule, and it is the second time this file records a version of it:**
a cost has to be measured in the units the user experiences. "0.56% of a step" and "1.2 s
before the first pixel" were the same code, and only one of those framings was ever checked.

### `anima-vk-bench`: int8 was never the problem

**`TensorPencil anima-vk-bench [<seq>]`** is the Vulkan counterpart of `anima-cuda-bench`,
and building it inverted the conclusion. Sync-per-op is exact here (outside `beginBatch`
every op ends in `submitAndWait`), and the harness prints bf16 and int8 **at identical
shapes** because the question was never "how fast is the int8 GEMM" but "is it faster than
the bf16 one it replaces, and by enough to pay for the prep".

At seq 6534, best-of-rounds, net of a measured 0.03 ms submit floor:

| | bf16 | int8 | ratio |
|---|---|---|---|
| q/k/v/out+cross `2048<-2048` x6 | 1.49 ms, **37.6 TF/s** | 0.74 ms, **77.5 TF/s** | 2.02x |
| mlp1 `8192<-2048` | 4.94 ms, 44.6 | 2.57 ms, 86.2 | 1.92x |
| mlp2 `2048<-8192` | 5.03 ms, 43.8 | 2.62 ms, 84.7 | 1.92x |

Per 28-block step: bf16 = gemm 529 + attn 699 + elt 266 = **1495 ms**; int8 = gemm 269 +
prep 83 + attn 699 + elt 266 = **1318 ms**.

⚠️ **So int8 is behaving exactly as it should and there is no int8 gap.** Its GEMMs are
**1.9-2.0x** the bf16 ones and run at **77-86 TFLOP/s — faster than cuBLASLt's bf16 (49-54)
on the same card.** The step only improves 1.13x because the GEMMs are ~35% of it while
**attention is ~47% and does not shrink**, and the prep adds back 24% of what the GEMMs
saved. That is arithmetic, not a defect — and it is the receipt that replaces the earlier
"Vulkan's int8 speedup is only 1.2x, cause unknown".

⚠️ **The qk-norm was 11.4x off, and the fix was a kernel already sitting in the file.**
`Elt.rmsnorm` gives each THREAD a whole 128-wide head, so a warp's 32 loads land 512 B apart
and every one is its own sector: **45 GB/s** at seq 6534 against `rmsnorm_sg`'s **511**. Three
calls per block x 28 blocks is **198 ms/step -> 18 ms**, i.e. the whole `elt` bucket drops
266 -> 86 ms. ⚠️ **Confirmed on a real render, which is what makes the bench trustworthy**:
1056x1584 / 30 steps / cfg 5 went **3.57 -> 3.11 s/step**, against a predicted 0.36 s/step
(180 ms x two forwards). Output quality is unchanged — **33.54 dB** against ComfyUI fp32
where it was 33.71, both far inside that render's own 23.04 dB bf16-vs-fp32 control. This is the CUDA `qk_rmsnorm_warp` finding repeating verbatim on the other
backend, which is the second time this file records it: **judge a bandwidth-bound kernel by
achieved GB/s against the card's roof, never by its share of the step.** At seq 1536 the same
kernel reads 176 GB/s and looks merely mediocre — the pathology only becomes obvious at
production size, because more rows means more independent thread-strided streams.
⚠️ **Not bit-identical**: the row sum becomes a subgroup tree where it was serial. It is the
more accurate of the two, and `ln_mod_sg` next door already reduces that way.

### The same fix across all three Vulkan DiTs

**`TensorPencil vk-norm-bench`** sweeps every weighted-RMSNorm shape the three Vulkan DiTs
use, at the sizes they render at, timing `Elt.rmsnorm` against `rmsnorm_sg`. It exists
because **the right choice is shape-dependent AND size-dependent** — thread-per-row is
catastrophic on a narrow row and merely bad on a wide one, and at a third of production size
Anima's own case understated the gap by 3x (176 vs 355 GB/s at seq 1536; 29 vs 509 at 6534).

| call site | rows x dim | thread/row | subgroup | | per-step estimate |
|---|---|---|---|---|---|
| anima Q/K @1056x1584 | 104544 x 128 | 29-37 GB/s | 462-509 | **12-18x** | 246 -> 19 ms |
| zimage Q/K @1056x1584 | 205440 x 128 | 36 | 508 | **14.2x** | 376 -> 27 ms |
| zimage sandwich (x4/block) | 6848 x 3840 | 108 | 315 | 2.9x | 250 -> 86 ms |
| krea2 Q/K **bf16/fp8 only** | 352800 x 128 | 32 | 550 | **17.1x** | 630 -> 37 ms |

All eight live sites now prefer the subgroup kernel (`zimage_gpu.rmsNorm`,
`dit_gpu.rmsNormQk`, `anima_gpu.qkNorm`).

**Confirmed by same-session A/B renders, which is the only number worth quoting:**

| | before | after | |
|---|---|---|---|
| Anima, 1056x1584 / 30 steps / cfg 5 | 3.57 s/step | **3.11** | -13% |
| Z-Image, 1056x1584 / 9 steps / cfg 1 | 4.44 s/step | **3.80** | -14% |

⚠️ **The table's estimate is an ESTIMATE, not a ceiling, and Z-Image beat it** — 640 ms
measured against 513 predicted. The row count in the sweep is approximate (Z-Image's joint
sequence at that resolution is larger than the 6848 used), so read those figures as an order
of magnitude for deciding *whether* to switch a site, and the A/B as the result. Quality is
unaffected: Anima **33.54 dB** against ComfyUI fp32 (was 33.71) and Z-Image **38.27 dB**
before-vs-after, both far inside their own precision floors (23.04 and 24.49 dB).

⚠️ **Two scoping facts I had to check rather than assume, and one of them killed a row of my
own bench.**

1. **krea2's Q/K `Elt.rmsnorm` is an `else` branch behind two FUSED alternatives**
   (`qknorm_rope16`, `qknorm_rope_f32`). `qkv_shared` requires `!is_bf16`, so **bf16 and fp8
   checkpoints fall through and do benefit, while an int8 one takes the fused path and never
   reaches it.** Quoting the 630 ms as a krea2 win generally would have been wrong.
2. ⚠️ **krea2's `dim`-wide block norms do NOT go through `Elt.rmsnorm` at all** — `dit_gpu`
   routes them through the already-parallel `rms_partial` -> `rms_combine` ->
   `rms_apply_mod` chain. My first bench listed a 6144-wide krea2 row implying a 262 ms
   saving; there is none. It is kept with `per_step = 0` and labelled
   `(ref) wide row, no such site`, precisely so the wide-row numbers cannot be read as an
   available win. **A bench row is a claim; an unlabelled shape is a claim about a call site
   that may not exist.**

**krea2 bf16 verified end to end** — the `else` arm renders its reference poster correctly on
Vulkan (768², text legible, no artefacts). ⚠️ **That render's 22.9 s/step says nothing about
the norm**: an 18.7 GB bf16 DiT on a 24 GB card is weight-streaming bound, so it is a
correctness check, and krea2's speed benefit remains **kernel-measured only** (32 -> 550
GB/s at its geometry). A clean krea2 speed A/B needs a card that holds the model resident.

⚠️ **Not bit-identical anywhere**: the row sum becomes a subgroup tree where it was serial
(the more accurate of the two, and `ln_mod_sg` already reduced that way). Z-Image's and
krea2's gated forward tests pin tolerances, so those are the things to watch, not just the
renders.

⚠️ **Vulkan's real gap against CUDA is ATTENTION, not the GEMMs.** 699 ms/step at **16.0
TFLOP/s** against `opAttnTC`'s **44.3** for the same work, and inside it the P@V (`out`) pass
is **14.93 ms against `md`'s 7.00** — 2.1x, on the half that should be the cheaper one. That
is where a 1056x1584 Vulkan render's 3.5 s/step sits, and it is orthogonal to quantization.

⚠️ **First version of this harness was unusable and the reason is on the record**: one
8-iteration mean, which moved **2.2x between runs** (bf16 mlp1 read 2.29 ms then 5.06;
`gelu_erf` 672 then 342 GB/s) because the 3090 idles its clocks. It now warms for 150 ms on
the op being measured and takes the **minimum over 5 rounds**. A short cold run on this card
is not a measurement.

⚠️ **Every activation buffer is 128-ROW PADDED, and it was not.** A quantized GEMM writes
`i8_mpad` rows (the prep pads m up to 128 and the coop tile is 128x128), where the dense
path writes exactly `m` — so the first int8 run died with
`CUDA_ERROR_ILLEGAL_ADDRESS`. `compute-sanitizer --tool memcheck` named it in one run:
`igemm_pipe_fused` writing 1 byte past a 1,572,864-byte (= 192 x 2048 x 4) allocation. **For
an illegal-address fault, reach for the sanitizer before forming a hypothesis** — I had
already spent a fix on a wrong one (`ensureDeviceBuffer` growing inside the batch, which
that function already guards against by syncing the stream, and which I then reverted).

**Validated by `TensorPencil anima-cuda-test [<ckpt>] [libs]`**, which checks the two new
kernels against the CPU ops they reproduce and then the whole device forward against
`anima.DiT.predict` on real weights, on both attention arms, exiting non-zero on failure;
plus the gated `-Dtest-filter="Anima gpu"` for the Vulkan arm.

### Container style, and what the resolver has to get right

⚠️ **Anima BUNDLES its VAE and ships its encoder separately — the opposite split from
Z-Image**, which is why `resolveComponent` deciding per component rather than per
family keeps paying off. The bundled `first_stage_model.*` is **byte-for-byte**
`qwen_image_vae.safetensors` (194 tensors, all shapes equal, verified), and
shape-for-shape krea2's Wan 2.1 decoder — so `Session.decode` shares krea2's whole arm,
including `latents_mean`/`latents_std`, and a test asserts the latent2rgb preview
resolves to krea2's matrix and NOT Z-Image's (also 16-channel, so the wrong one is not
even an out-of-bounds read).

⚠️ **`defaultComponentPath(.anima, .decoder)` is deliberately EMPTY.** A defaulted path
that does not exist is not an error — the resolver falls through to the checkpoint's own
copy — but naming one would send it at a file that has nothing to do with this family,
which is the failure a joined SD1.5 checkpoint already hit once.

⚠️ **`detectFamily` probes the LLM ADAPTER, not the trunk**, and `anima_probe` is one
constant shared with `componentSpec` so the two cannot disagree. Every trunk tensor name
— `blocks.0.mlp.layer1.weight` included — is shared with stock Cosmos-Predict2, so a
trunk probe would load a Cosmos checkpoint as Anima and fail on ~100 missing adapter
weights. The detection test's Anima cases carry the Cosmos trunk tensor **as well**, so
they pin that the adapter is what decides, and a negative case asserts a Cosmos trunk
*without* an adapter is `UnknownArchitecture`.

⚠️ **`net.` is a third denoiser prefix, and omitting it made the BASE checkpoint
unloadable (fixed 2026-08-07).** `anima_baseV10.safetensors` stores all 685 tensors
under `net.`; the prefix list was `{"model.diffusion_model.", ""}`, so `detectFamily`
found no adapter and the render died with **`error.UnknownArchitecture`** — a checkpoint
ComfyUI opens fine, refused with an error that names the *architecture* when the problem
was the *prefix*. It is ComfyUI's own third candidate in `unet_prefix_from_state_dict`,
where the comment against it reads `# cosmos`, and Anima **is** Cosmos-Predict2's DiT.
- The prefix list was **duplicated** between `detectFamily` and `componentSpec`, which is
  what let the two disagree: the resolver would have opened the file that detection had
  already refused. `detectFamily` now reads `(componentSpec(fam, .denoiser)).prefixes`
  for both Anima and Z-Image, so a new spelling is added in one place.
- Verified end to end: that checkpoint now renders. ⚠️ It is **denoiser-only** (no
  bundled VAE), so it needs an explicit `--vae` — and the resolver says exactly that
  (`decoder not found: the checkpoint has no 'decoder.conv1.weight' under any known
  prefix`) rather than failing deep in a weight load.

⚠️ **The gated Anima real-checkpoint tests had been silently SELF-SKIPPING**, because
`anima_ckpt` names the checkpoint the fixture was generated from (tied to it by sha256,
correctly) and that file is no longer on the box. So "17/17 green" meant 17 tests of
which the real-weight ones never ran. Nothing is wrong with the gating — that is what
`requireModelFile` is for — but **a green summary is not coverage**, and the fix for the
one case that needed no real weights was to make it synthetic and ungated rather than to
repoint the shared constant, which would have compared against the wrong weights.

**Defaults**, from ComfyUI's own `image_anima_base_v1` template: 1024², **30 steps**,
**cfg 4.0**, euler + **`simple`**, shift 3.0, with `qwen_3_06b_base.safetensors` as the
encoder. (The reference render this was validated against used cfg 5 and `normal` —
the user's own choice, not a default.)

### The per-image stall: 13.7 s -> 1.3 s, and why the bench understated it 3x

Reported as "all anima images are super slow to actually start … i don't see this kind of
delay with other models, it's just this anima one", with `[anima] per-image device session
built in 12.21s` **repeating on every image** and high CPU throughout.

Cause: `DiT.modulationSchedule` built the AdaLN table **one sigma at a time**, calling
`modulationTable` per sigma. At 28 blocks x 3 sublayers that is **5270 GEMV calls with
m = 1** per image — 11.96 GFLOP with no weight reuse. Batched over the sigma axis it is
**170 GEMMs with m = n_sigmas**, the same arithmetic with each weight read once.
**Measured** (Debug, 1024², 30 steps, cfg 5, `cuda`, isolated with a temporary env switch
over exactly that branch): **13.74 s -> 1.28 s**.

⚠️ **The first bench of the two forms said 3.5x, understating the real win by 3x, and the
reason generalizes.** It ran on `testing.io`, which has no thread pool. Most of what
batching removes is not arithmetic but **task-spawn overhead**: `ops.matmul` splits into
`4 * n_threads` tasks whenever an op clears 1 MFLOP, which every one of these tiny GEMVs
does, so the unbatched form paid **~168,000 task spawns per image**. An Io with no pool
cannot see the dominant cost. Same rule as the profiler-bucket lesson: **measure in the
units the caller experiences**, and check what the harness actually exercises.

⚠️ **Two wrong diagnoses first, both from ratios rather than isolations.** (1) The host
cross-KV projection, correctly sized at 0.56% of a *step* — a share of the wrong quantity,
since this is a one-off before the first step. (2) The **Debug build**, which is real
(measured **10.7x** on this path: 0.96 s vs 0.09 s at 256²) and is why the user found it
"barely noticeable in ReleaseFast" — but Debug alone accounts for 1.24 s of 13.74 s, so it
was a multiplier on the defect, not the defect. Only the A/B over the branch settled it.

⚠️ **Bit-identical below 16 sigmas, last-bits different at or above it**, because `m`
crosses `ops.matmul.small_m_max`: under 16 a row still goes through ggml's `vec_dot`
exactly as the m = 1 form did, at 16 and up it takes the packed f32-panel path with a
different reduction order. Verified in **both** regimes — **PSNR inf (bit-identical)** at
8 steps / cfg 1, and **46.8 dB** at 30 steps / cfg 5, the latter being trajectory
amplification of a last-bits input change and **24 dB inside this model's own measured
cfg-5 precision envelope** (ComfyUI disagrees with itself by 23.04 dB there). A single
render comparison would have read 46.8 dB as a defect; the cfg-1 control is what says it
is not. Pinned by an **ungated** synthetic-weight test asserting rel L2 < 1e-6 against
`modulationTable` at five sigmas spanning the schedule.

### What the width generalization cost

⚠️ **`qwen3.TextEncoder.encode` and both GPU twins were hardcoded to the 2560-wide
4B body** (`const hidden = qwen3.hidden`), so a 1024-wide Qwen3-0.6B ran with 2.5x
strides — finite garbage, no error. All three now read `enc.cfg`. `Variant.anima` also
needed a new tap KIND: krea2 and Z-Image tap *before* a layer that still has to run, so
their final norm is genuinely never evaluated and is not even loaded, while Anima taps
one past the last layer (`layer = "last"`, `final_norm = True`) and the norm always
applies. A fast test asserts the invariant both ways — the final norm is loaded iff the
last tap is `n_layers` — since a tap list that silently stops agreeing with its config
is a finite encode of the wrong hidden state.

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

## Tool calling: the model's own trained format, both directions

Landed 2026-08-07. Before it, **no model in TP could be given tools at all**: the jinja
engine could render the trained format correctly (that was fixed while landing Bonsai-27B —
`tojson`'s separators and the missing `items` filter), but nothing could feed it.
`chat_template.zig` had no `tools` global, no `Message.tool_calls`, and `Role` had no
`.tool`.

Four pieces, plus the read half:

| | |
|---|---|
| `RenderOpts.tools: ?[]const Tool` | → the template's `tools` variable |
| `Message.tool_calls` / `.tool_call_id` / `Role.tool` | an assistant call turn and its answer, replayable |
| `llm/tool_call.zig` | parses the calls a model EMITS, back into that typed form |
| `tp-llm --tools <file.json>` + `/tool <result>` | a real round trip, driven by hand |
| `chat_template.parseTools` | an OpenAI-style declaration (bare or `{"type":"function",…}`-wrapped) |

**Verified end to end** on Bonsai-27B (`--backend cuda`, `--tools` with one `get_weather`):
the declaration renders into the system prompt, the model emits
`<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>`,
that parses back to `get_weather({"city": "Paris"})`, `/tool 18C and sunny` replays as
`<|im_start|>user\n<tool_response>\n18C and sunny\n</tool_response>`, and the model answers
from it.

⚠️ **The wire format is NOT JSON on the models here.** qwen3.5/Bonsai are trained on the
XML-ish form above (their own template documents it in the system prompt), while plain
qwen3 and most llama finetunes emit the Hermes `{"name":…,"arguments":{…}}` body. `parse`
auto-detects on the first non-space byte, because a session cannot know which it will get.

⚠️ **`arguments` is a DICT, not a string.** The templates iterate it (`arguments | items`)
to emit one `<parameter=…>` block per key. Handing over OpenAI's wire shape — the arguments
as a JSON *string* — renders one `<parameter=…>` containing the whole JSON blob. The golden
test's argument values (bool, null, dict, list, int, string) are exactly the ones where the
template's own `args_value` spellings disagree, so this cannot pass by accident.

⚠️ **Tool support is a property of the TEMPLATE and is MEASURED, never inferred from the
architecture** (`supportsTools` / `supportsToolRole` render a probe transcript). The four
templates in the golden corpus land in three different places: qwen3.5/Bonsai and gemma4
do both, plain llama echoes a tool turn but declares nothing, and **gemma3 raises**. A name
list would additionally be wrong for the first finetune that strips its tool branch.

⚠️ **gemma3 raises on an INSERTED TURN OF ANY ROLE**, not just on `tool` — its template
asserts strict user/assistant alternation. So "just slip the result in as an extra user
message" is not a safe fallback: it kills every render from that point on, and the failure
is a dead render rather than a degraded one. Anything that inserts a turn the user did not
type has to reckon with this; pinned by a test that renders the alternating baseline *and*
both bad variants.

**The two halves are pinned as INVERSES**, not just individually: a `ToolCall` rendered
through Bonsai's real template, parsed back by `tool_call.parse`, returns the same name and
the byte-identical `arguments_json`. A drift there means the model sees a garbled version
of its own request one turn later — which no shape check would catch.
- ⚠️ The inverse is genuinely **ambiguous for a string that looks like JSON**: an argument
  whose value is the text `3` comes back as the number 3. The format carries no types and
  the template renders the two identically, so no reader can distinguish them; the tool
  schema is what a consumer has to rely on. Recorded rather than papered over.
- Value delimiting strips **exactly one newline per side**, the exact inverse of the
  template's `'\n' + value + '\n'` — a `std.mem.trim` would eat the leading indentation of
  a multi-line value, which the format explicitly supports.

**`splitThought`/`answerText`/`endsInsideThought` moved out of `gui/toolcall.zig` into
`llm/tool_call.zig`.** Both scanners need "where does the answer start", and ⚠️ two
definitions of it is exactly the drift that lets one caller fire a tool the other hides —
the `primed` bug this file already records once. `chat.Reasoning` now aliases
`tool_call.Reasoning` so a family's markers pass through with no conversion step.

### The GUI's `<image>` tool: a tool RESULT was built, measured against the product, and removed

⚠️ **Recorded because the code was right and the feature was still wrong, and the next
person to read TODO #2 will be tempted to rebuild it.** The stated defect is real — the
`<image>` tag is prompted and fire-and-forget, so nothing returns to the model and it
cannot tell "generated" from "OOM"; it will say "here's your image!" when there is none.
A per-call outcome turn (generated + size/seed, FAILED + reason, canceled, still
generating), inserted after the assistant turn that made the calls, was built and unit
tested. It was then taken back out.

**Why, and the reasoning generalizes to any tool in a conversational UI:**

- **A tool result earns its keep when the model must act on it to continue.** This one
  never continues: it emits the tag, the turn ends, and the next event is the *user*
  looking at the screen. The human is in the loop on every single turn and can see the
  image, or its failure, directly — they have strictly more information than the model.
- **It cannot fix the sentence it was meant to fix.** "Here's your image!" is written in
  the SAME turn as the call, before generation starts. No result can retract it; it only
  lets the model be correct one turn later, by which point the user already knew.
- ⚠️ **A result turn is not free on a small local model.** `<tool_response>` for a tool
  the system prompt never declared invites narrating the report at the user; the
  no-tool-role fallback (folding it into the assistant's own turn) invites *imitating* the
  marker, i.e. fabricating outcomes — which is the exact lie being fixed.
- **Identification was the concrete defect**: results were keyed by an ordinal over the
  turn's calls, so the model had to count its own `<image>` tags to match one. The natural
  key is the prompt text the model itself wrote. Worth knowing if this is ever revisited.
- Minor: a pending→done transition rewrites the middle of the transcript, re-prefilling the
  tail.

**What was kept, because the audience was wrong rather than the information:**
`GenImage.gen_error` / `failure()` records *why* a generation failed (same `@intFromError`
shape as `Diffuser.load_error`); it was previously only `std.log.err`'d, i.e. visible to
nobody but a terminal. It belongs to the **user**, not the model — which is TODO #3's error
message and retry button, now unblocked. ⚠️ `fail()` stores the reason **before** the
status: the UI thread learns of a failure by polling `status`, so the other order lets it
read a fresh `.failed` beside a stale zero reason.

**The capability that would actually be worth it** is the other half of TODO #2 — handing
the generated image back **as an image** so a multimodal model can look at what it made.
Note that the user-driven form already exists: the "discuss this image" button attaches a
generated image to the next user turn (`attachOrStageRgba`, gated on vision capability).
So the open question there is only whether it should ever be automatic, not whether the
plumbing exists.

### A failed image says why, and retries in place

`GenImage.gen_error` (above) is consumed by both image surfaces — the chat transcript and
the studio gallery — which previously rendered a bare `⚠ failed`.

⚠️ **VRAM exhaustion reaches the UI under FOUR different error names**, so mapping only
`OutOfMemory` would leave the single most common failure looking like an internal bug: the
cuBLASLt/cuDNN libraries report an out-of-workspace as their own error, and the hand-PTX
path surfaces a post-OOM fault as `CudaError`. `diffuser.failureText` collapses that family
to "out of VRAM" and names the other actionable cases; anything unrecognized falls back to
the error name, which is at least specific enough to search for. Same set
`pipeline.recoverableDecodeErr` already had to enumerate — worth checking both if a new
device error appears.

**"Try again" re-queues the image IN PLACE** (`Diffuser.retry`): the `*GenImage` is
unchanged, so a chat variant's borrowed pointer turns the same tile back into a progress
bar instead of a second image appearing below the first. Two deliberate inversions:

- ⚠️ **The model snapshot is REFRESHED to the live config**, inverting `enqueue`'s rule that
  an image finishes on the config it was created with. That rule protects a queue from a
  mid-run switch; a retry is a fresh request made *now*, and the whole reason to press it
  after an OOM is that the user just changed something. Honouring the old snapshot would
  retry the exact configuration that failed.
- ⚠️ **The resident pipeline is dropped first** (when nothing is in flight), which is what
  makes the retry a different attempt rather than a replay: after a VRAM failure the
  resident session is holding the memory the retry needs. **NOT verified to clear a sticky
  CUDA fault** — `CUDA_ERROR_ILLEGAL_ADDRESS` poisons its context and whether a session
  rebuild escapes that has not been measured here; if it does not, the retry reports the
  same error again, which is at least honest.

A stale `resume_snapshot` is dropped too — resuming a retry from some earlier pause's latent
would start mid-render into a run that has nothing to do with it.

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
