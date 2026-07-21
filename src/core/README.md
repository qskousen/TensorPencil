# tp_core — foundational primitives

The bottom of the stack: pure-Zig data and primitive types with **no dependency
on any compute backend, model, or pipeline**. Everything above imports this by
name (`@import("tp_core")`); nothing here imports upward.

Root: `core.zig`.

## Public surface (`tp_core.<name>`)

| Namespace | What |
|---|---|
| `dtype` | `DType` and dtype metadata (sizes, block-quant layout) |
| `tensor` | `Tensor`, `Shape`, `TensorInfo` |
| `safetensors` | `SafeTensors` reader (`.open` / `.openIn`) |
| `gguf` | `Gguf` reader |
| `quants` | block-quant dequant (`dequantSlice`, ggml-backed); `have_ggml` |
| `weights` | `WeightStore` over a checkpoint |
| `tokenizer` | `Tokenizer` (embedded qwen vocab under `assets/`) |
| `kv_cache` | `KvCache` / `PerLayerKvCache` autoregressive caches |
| `sample` | logits→token: argmax, top-k/top-p/min-p, penalties |
| `sampler` | diffusion scheduler; `torch_rng` Torch-compatible RNG |
| `image` | image load/save |
| `spec_limits` | speculative-decode size caps; `prof` profiling |

## Dependencies

None (std only). ggml is linked in when `-Dggml` is on (default) to back the
block-quant paths; with `-Dggml=false` those return `error.QuantBackendUnavailable`.

## Minimal consumer example

```zig
// build.zig
const tp = b.dependency("TensorPencil", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("tp_core", tp.module("tp_core"));
```

```zig
// src/main.zig — read a checkpoint header, no GPU/model code compiled in
const core = @import("tp_core");
var st = try core.safetensors.SafeTensors.open(gpa, io, "model.safetensors");
defer st.deinit();
std.debug.print("{d} tensors\n", .{st.index.count()});
```
