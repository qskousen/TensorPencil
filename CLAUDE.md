# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Ground Rules

- Never run `git add` or `git commit` unless directly requested.
- `zig build` produces no output on success; any output indicates a warning or error.
- **Read `ZIG.md` before doing any work.** It documents Zig 0.16.0 breaking changes relevant to this codebase. When you encounter and resolve a new 0.16.0 change, add it to `ZIG.md`.
- We want the code to be clean, with clear seperation of concerns, modular, and testable.
- If there is ambiguity in a request, don't guess or assume; ask for clarification.
- When adding a new feature or fixing a bug, add unit / integration tests as appropriate.
- Make sure all tests are still passing after working on something. If they aren't, fix it - even if the test was previously broken.
- Remember that the best code is often the simplest code. Tend towards simple solutions where possible, that will work for all the edge cases.
- If the user asks for something that may cause issues, push back and get confirmation before doing it.
- If you see existing code that may cause issues or is Band-Aid patch code, call it out and suggest a fix.
- This is a research project. There's no risk to trying big complicated work. We want to try unusual things. Be bold and adventerous.

## Project

TensorPencil is a diffusion inference engine (text-to-image) plus an LLM inference engine (`tp-llm`, see `LLM_PLAN.md`) written in pure Zig, targeting **Zig 0.16.0** (`minimum_zig_version` in build.zig.zon). "Pure Zig" means no C dependencies or bindings — tensor ops, model loading, tokenization, samplers, and image encoding are all to be implemented in Zig.

The repo is currently the stock `zig init` scaffold; the engine has not been built yet.

## Commands

- `zig build` — build the executables (installs to `zig-out/bin/TensorPencil` and `zig-out/bin/tp-llm`)
- `zig build run -- <args>` — build and run the diffusion CLI with arguments
- `zig build run-llm -- <args>` — build and run `tp-llm`, the LLM inference CLI (see `LLM_PLAN.md`)
- `zig build test` — run all tests (both the library module and executable module test binaries)
- `zig build test --fuzz` — run fuzz tests (`std.testing.fuzz`)
- `zig build -Doptimize=ReleaseFast` — optimized build (important for benchmarking inference; Debug is very slow for numeric code)

To run a single test, filter at the compiler level since `zig build test` runs everything:
- `zig test src/root.zig --test-filter "<test name substring>"`

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

No build-time or Zig-package dependencies, and the goal is to keep it that way (pure Zig). If one ever becomes necessary, add it with `zig fetch --save <url>` (populates `build.zig.zon`).

Runtime `dlopen`'d system libraries are allowed (no linking, no headers, no nvcc — device IR is hand-emitted), gated per backend:
- Vulkan loader (`libvulkan.so.1`) — `--backend vulkan`.
- CUDA driver (`libcuda.so.1`) — `--backend zig-cuda` (hand-emitted PTX; still "pure Zig" by the project's standard).
- **`--backend cuda` deliberately crosses the pure-Zig line**: it `dlopen`s NVIDIA's closed-source math libraries `libcublasLt.so` (int8/f16 GEMM) and `libcudnn.so.9` (fused SDPA attention + conv). This is the one non-pure backend, added to measure the "our kernels vs their libraries" gap (M10 Phase 2). The CPU / Vulkan / zig-cuda backends stay pure and are the default; keep it that way when adding features.

One build-time C linkage exists, contained to the **tp-llm executable** (the TensorPencil library module stays pure Zig): image DECODE for `--image` / chat `@mentions` goes through system **libvips** via a tiny C shim (`lib/vips/vips_helper.c` + `src/vips.zig`, ported from DiffKeep) — jpeg/png/webp/gif/tiff input with EXIF rotation. Building needs `libvips-dev` + pkg-config. The library's own pure-Zig PNG encode/decode in `image.zig` is unaffected (diffusion output, tests). Don't spread vips into the library module.
