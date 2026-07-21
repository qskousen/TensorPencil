# tp_gpu — GPU backends

The device compute layer: a pure-Zig Vulkan backend (runtime-loaded loader +
Zig-authored SPIR-V kernels, embedded at build time) and an experimental CUDA
driver-API backend (dlopen'd `libcuda` + hand-emitted PTX, plus the optional
cuBLASLt/cuDNN path). No model or pipeline code is imported here.

Root: `../gpu.zig`.

## Public surface (`tp_gpu.<name>`)

| Namespace | What |
|---|---|
| `Context` / `context` | Vulkan device context (`Context.init`, `.matmul`, …) |
| `vk` | Vulkan loader bindings |
| `spv` | SPIR-V kernel module helpers |
| `cuda` | CUDA driver-API backend (`cuda.Backend`, …) |
| `mem_tag` | device-allocation tagging (`MemTag`) |

The SPIR-V kernels (`matmul_f8`, `matmul_f32`, `transpose`, `eltwise`) are
compiled to `.spv` by the self-hosted backend and embedded via
`@embedFile("<name>_spv")`.

## Dependencies

`tp_core`, `tp_ops`. Device libraries are `dlopen`'d at runtime, so linking this
module needs no CUDA/Vulkan SDK — only libc for the loader.

## Minimal consumer example

```zig
// build.zig
const tp = b.dependency("TensorPencil", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("tp_gpu", tp.module("tp_gpu")); // pulls tp_core+tp_ops transitively
```

```zig
const gpu = @import("tp_gpu");
const ctx = gpu.Context.init(gpa) catch {
    std.debug.print("no Vulkan device\n", .{});
    return;
};
defer ctx.deinit();
std.debug.print("device: {s}\n", .{ctx.deviceName()});
```
