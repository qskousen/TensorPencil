# Using TensorPencil as a library

TensorPencil is not only two CLIs (`TensorPencil`, `tp-llm`) and a GUI (`tp-gui`) —
its engine is a reusable Zig module. The binaries are thin drivers over it;
anything they do, a consumer can do.

The engine is split into a stack of **independently-consumable layer modules**,
so you can depend on just the tier you need — the raw primitives, the GEMM
kernels, the GPU backends, the offload logic, or the whole thing:

| Module | Layer | Depends on |
|---|---|---|
| `tp_core` | dtypes, tensors, checkpoint parsing (safetensors/GGUF), quant **quantize + dequant**, tokenizer, K/V cache, sampler | — (pure Zig) |
| `tp_ops` | CPU numeric kernels: `matmul` (GEMM), `attention`, `rope`, `norm`, `act`, `convrot` | `tp_core` |
| `tp_gpu` | GPU backends: Vulkan + embedded SPIR-V kernels, CUDA driver-API PTX | `tp_core`, `tp_ops` |
| `tp_runtime` | offload/scheduling: the VRAM arbiter + CPU/GPU residency planner | — (pure Zig) |
| `tp_models` | NN architectures (DiT/VAE/qwen/gemma/ViT) + LLM generation (chat/engine/spec) | all of the above |
| `TensorPencil` | the umbrella: the diffusion pipeline + re-exports of every layer | all of the above |

### Side effects a `tp_gpu` consumer should know about

`gpu.Context.init(gpa, io)` takes an `Io` because 0.16 puts file access and clocks behind
one, and it uses both: on a GPU it has no measurement for, it **screens the cooperative-matrix
warp tiles once and saves the result** under the user's cache dir, keyed by
`vendor:device:driver`. Later runs on the same device+driver just load it; a cache written by a
different GPU is ignored. It costs 0.3s on an RTX 3090 and 13s on an Arc A310, once per machine,
and buys ~4x end-to-end on a device whose hardware fragment shape differs from NVIDIA's.
`pipeline.Session.init` inherits this, so neither needs an extra call.

Two knobs for embedders that cannot accept a first-run cost or a write outside their own state
directory (`BACKEND.md` has the detail):

```zig
tp.gpu.context.autotune_tiles = false;         // never screen; use the default tile
tp.gpu.context.tile_cache_path = "/my/state";  // keep the measurement somewhere else
```

Each is exposed via `dep.module("<name>")`. Importing a lower module compiles
only that tier and its dependencies — e.g. pulling `tp_ops` for the GEMM kernels
does not drag in models or the pipeline. The umbrella `TensorPencil` re-exports
everything (`tp.ops`, `tp.gpu`, `tp.models`, `tp.pipeline`, …) for consumers who
just want it all.

## Requirements

- Zig **0.16.0** (`minimum_zig_version`).
- A C/C++ toolchain reachable by Zig (for the optional ggml dependency; see below).
- Nothing else at build time. GPU backends (`libcuda`, `libvulkan`, cuBLASLt,
  cuDNN) and image decode (`libvips`) are `dlopen`'d / linked only by the
  executables, not the library module — the module itself stays pure Zig.

## Depending on it

From your project root:

```sh
zig fetch --save "git+https://github.com/<owner>/TensorPencil#<commit>"
```

Then in your `build.zig`:

```zig
const tp = b.dependency("TensorPencil", .{
    .target = target,
    .optimize = optimize,
    // .ggml = false,   // opt out of the ggml block-quant backend (see below)
});

const exe = b.addExecutable(.{
    .name = "myapp",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true, // the module dlopen's system libs, so it links libc
        .imports = &.{
            .{ .name = "TensorPencil", .module = tp.module("TensorPencil") },
        },
    }),
});
```

The module carries its own transitive pieces (the SPIR-V compute kernels are
embedded; the ggml static lib + libc++ are linked in automatically when
enabled), so you only wire the one import.

## The `-Dggml` option

ggml (llama.cpp's tensor library) backs the **GGUF block-quant** paths
(`q4_0`/`q8_0`/`q4_k`/`q5_k`/`q6_k`): dequantization and the small-batch quant
GEMV. It is a fetched dependency, on by default.

- **`-Dggml=true` (default):** full capability. ggml is fetched and its AVX2 CPU
  quant kernels are linked in.
- **`-Dggml=false`:** ggml is neither fetched nor compiled. Block-quant dtypes
  become unavailable — `ops.matmul.matmul` returns
  `error.QuantBackendUnavailable` for them, and `quants.dequantSlice` panics.
  Everything else is unaffected: `f32`/`f16`/`bf16`/`fp8`/`int8`/`int4` weights,
  safetensors models, every GPU kernel, and the whole diffusion path still work.

Query it at runtime via `TensorPencil.quants.have_ggml`.

A pure-Zig fallback for block-quant (so quantized models work without ggml, just
slower) is not implemented; it would be a separate piece of work.

## Quantization (`quants`)

`quants` is not dequant-only — it is the ggml **format** layer, in both directions,
so a consumer that writes checkpoints (e.g. a converter) does not need its own ggml:

- `quants.dequantSlice(dt, row, elem0, n, dst)` — block-aligned dequant for a
  `DType` (golden-fixture validated against llama.cpp's reference decoders).
- `quants.quantizeChunk(dt, src, dst, nrows, n_per_row, imatrix)` — f32 → block
  quant for a `DType`.
- `quants.raw.*` — the same operations plus layout/metadata queries keyed on the
  **numeric `enum ggml_type` id** rather than `DType`. `DType` covers the formats TP
  can *compute* with; `raw` reaches every type ggml knows, including ones we have no
  kernel for and deliberately do not model. That is what a tool inspecting or
  rewriting arbitrary GGUF files needs: `raw.name`, `blockElems`, `blockBytes`,
  `rowBytes`, `isQuantized`, `requiresImatrix`, `ensureQuantizeInit`,
  `quantizeChunk`, `dequantRow`. Out-of-range ids, missing kernels and unaligned
  lengths come back as errors rather than C-level undefined behaviour.

The `imatrix` argument is ggml's per-column importance weighting (one weight per
column, shared across the rows in a call): the k-quant and IQ scale searches
minimize *weighted* error against it instead of plain squared error. Pass
per-channel activation statistics there for activation-aware quantization.

Note both directions are only as reproducible as the ggml build: the `q6_k`, `q5_k`
and `q2_k` scale searches are sensitive to floating-point contraction, so their
output bytes differ between optimization levels. TP always builds ggml at
`ReleaseFast` regardless of the consumer's `-Doptimize`, which keeps this stable.

## What's reachable

Everything public is re-exported from the module root (`src/root.zig`). The
main namespaces, low to high:

| Namespace | What |
|---|---|
| `dtype`, `tensor` | dtypes, tensor/shape primitives |
| `safetensors`, `gguf`, `quants`, `weights` | checkpoint parsing + weight stores (`WeightStore` unifies the two; the DiT and every LLM load through it, and `weights.Overlay` substitutes individual tensors from memory) |
| `ops` | CPU numeric kernels: `matmul`, `attention`, `rope`, `norm`, `act`, `convrot`, `conv`, `vmath` |
| `gpu` | backends: `gpu.vk` (Vulkan + SPIR-V), `gpu.cuda` (Driver-API PTX), `gpu.Context` |
| `vram` | the VRAM arbiter / budgeting |
| `models` | architectures (qwen3/3.5, gemma3/4, DiT, VAE, ViT towers, `lpips`, …) + `models.residency` (CPU/GPU offload) |
| `tokenizer`, `sampler`, `image` | tokenization, sampling, image I/O + comparison metrics (`psnr`, `ssim`, `detailEnergy`) |
| `llm` | LLM orchestration: chat templating, generation `engine`, `kv_cache`, speculative decode |
| `pipeline` | the end-to-end diffusion pipeline, and the composable stages it is built from (`Session.encode` / `schedule` / `Session.denoiser` + `predict` / `Session.decode`), plus `Session.replaceDit` to swap the diffusion weights of a live session |

## Minimal example

```zig
const std = @import("std");
const tp = @import("TensorPencil");

pub fn main() !void {
    // e.g. read a safetensors header
    var st = try tp.safetensors.SafeTensors.open(allocator, io, "model.safetensors");
    defer st.deinit();
    std.debug.print("{d} tensors\n", .{st.index.count()});
}
```

See `src/main.zig` (diffusion) and `src/llm_main.zig` (LLM) for full drivers
exercising the pipeline, offload, and backend selection.
