# tp_runtime — offload / scheduling

The memory-management tier: VRAM budgeting and the CPU/GPU layer-residency
planner. This is **pure logic** — it computes budgets and offload *plans*; the
actual device allocations and host↔device copies happen in the model backends
that consume those plans. Because it holds no device handles, it has no compile
dependency on `tp_gpu` (it sits above it only conceptually).

Root: `runtime.zig`.

## Public surface (`tp_runtime.<name>`)

| Namespace | What |
|---|---|
| `vram` | the VRAM `Arbiter` — split/limit budgeting across resident models |
| `residency` | CPU/GPU layer-split planning (`growPlan`, offload grow steps) |

## Dependencies

None (std only).

## Minimal consumer example

```zig
// build.zig
const tp = b.dependency("TensorPencil", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("tp_runtime", tp.module("tp_runtime"));
```

```zig
const runtime = @import("tp_runtime");
// Plan how many decoder layers to keep resident under a VRAM budget, offloading
// the rest to host — the numbers only; the caller performs the moves.
_ = runtime.residency;
_ = runtime.vram;
```
