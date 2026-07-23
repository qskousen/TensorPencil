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
- **Default to `zig build test` (fast, ~15s CPU unit suite). Do NOT run `zig build test -Dintegration` unless you truly need it** — it runs the GPU device tests and real-model inference tests and takes ~11 minutes. Reach for `-Dintegration` only when your change touches GPU kernels / device code or the real-model LLM/parity paths, and even then prefer narrowing with `-Dtest-filter "<substring>"`. See the Commands section for the full split.
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
- `zig build test -Dtest-filter "<substring>"` (add `-Dintegration` for a gated test), or `zig test src/root.zig --test-filter "<substring>"` for a standalone compile.

## Architecture

Two-module layout wired in `build.zig`:

- **`src/root.zig`** — root of the public `TensorPencil` module (created with `b.addModule`, importable by consumers). All engine code (tensors, model/weights loading, schedulers/samplers, text encoder, UNet/DiT, VAE decode, image output) belongs under this module; anything public must be re-exported from `root.zig`, since consumers can only reach declarations visible from the module root.
- **`src/main.zig`** — root of the unnamed executable module; the CLI. It imports the library via `@import("TensorPencil")` and should stay a thin argument-parsing/driver layer over the library.

Tests live in both modules; `zig build test` builds and runs two separate test executables (one per module) in parallel.

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
