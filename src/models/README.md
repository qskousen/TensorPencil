# tp_models — architectures + LLM generation

The model tier. Two halves live in one module because they are mutually
recursive and neither depends on the diffusion pipeline above:

- **`models`** (`src/models/`) — the neural-net architectures: text encoders,
  DiT, VAE (+ tiled/GPU/CUDA variants), the qwen3 / qwen3.5 / gemma3 / gemma4
  LLMs, the ViT towers, and the EAGLE-3 drafter. Each has CPU + `_gpu` (Vulkan)
  + `_cuda` variants where implemented.
- **`llm`** (`src/llm/`) — generation orchestration that drives any model by
  duck typing: chat templating (`chat`), the decode loop (`engine`), speculative
  decoding (`spec`), the REPL (`repl`), and the session wrapper (`session`).

Root: `../tp_models.zig` (re-exports `models`, `llm`, and the `test_gate`
integration-suite gate). They share a module so `spec` can reference a concrete
model, `eagle3` can draft for qwen3, and the model integration tests can drive
the engine — all as intra-module references.

## Public surface

`tp_models.models.<arch>` (e.g. `models.qwen3.CausalLM`, `models.dit`,
`models.wan_vae`) and `tp_models.llm.<part>` (e.g. `llm.engine.generate`,
`llm.chat`, `llm.spec`).

## Dependencies

`tp_core`, `tp_ops`, `tp_gpu`, `tp_runtime`.

## Minimal consumer example

```zig
// build.zig — the whole model+generation tier without the diffusion pipeline
const tp = b.dependency("TensorPencil", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("tp_models", tp.module("tp_models"));
```

```zig
const tpm = @import("tp_models");
const core = @import("tp_core");
var st = try core.safetensors.SafeTensors.open(gpa, io, te_path);
defer st.deinit();
var lm = try tpm.models.qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
defer lm.deinit();
// ... drive it with tpm.llm.engine.generate(...)
```
