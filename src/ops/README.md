# tp_ops — CPU numeric kernels

The compute primitives shared by every model: dtype-aware GEMM, attention,
normalization, activations, RoPE, and ConvRot. Pure CPU — the GPU backend is
never imported here; large GEMMs are routed to a device only through an
injected hook (`matmul.gpu_dispatch`, set by the pipeline), so this layer stays
backend-free.

Root: `../ops.zig`.

## Public surface (`tp_ops.<name>`)

| Namespace | What |
|---|---|
| `matmul` | `matmul(io, gpa, y, x, m, w, bias)` — `y = x @ Wᵀ (+bias)`, dtype-aware (`Weight`), block-quant via ggml; `gpu_dispatch` injection hook |
| `attention` | attention (+ `attentionTree`) |
| `norm` | `layerNorm`, RMSNorm |
| `act` | `silu` / `geluTanh` / `sigmoid` + fused `*Mul` |
| `rope` | rotary position embedding |
| `convrot` | ConvRot (group-wise Hadamard) un-rotation |
| `vmath` | SIMD helpers (`expVec`, …) |

## Dependencies

`tp_core`. ggml (optional, `-Dggml`) backs the small-batch block-quant GEMV.

## Minimal consumer example

```zig
// build.zig — pulling just the GEMM kernels + primitives (no models/pipeline)
const tp = b.dependency("TensorPencil", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("tp_core", tp.module("tp_core"));
exe.root_module.addImport("tp_ops", tp.module("tp_ops"));
```

```zig
const core = @import("tp_core");
const ops = @import("tp_ops");

var w = [_]f32{ 1, 0, 0, 1 }; // 2x2 identity
const weight = ops.matmul.Weight.fromF32(&w, 2, 2);
var x = [_]f32{ 3, 4 };
var y: [2]f32 = undefined;
try ops.matmul.matmul(io, gpa, &y, &x, 1, weight, null); // y == {3, 4}
_ = core;
```
