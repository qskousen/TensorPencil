## TensorPencil

**This is an experimental work in progress.**

TensorPencil is a test to see how far we can push diffusion performance in pure Zig (aside from Vulkan / CUDA libraries).

It currently targets FP8 and INT8 ConvRot Krea 2, and those are the only models that have been tested.
Currently the Vulkan results differ slightly from ComfyUI but are pretty close - the Zig noise kernel was created to generate bit-identical initial noise,
and any remaining difference is due to the Zig DiT kernels reducing in a different order than cuBLAS/FlashAttention;
over 20 steps that difference accumulates as slight texture-level drift.

|                                 ComfyUI                                  |                                    TensorPencil                                    |                                                                                                                                                                               
|:------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|
| ![ComfyUI reference](testdata/comfyui_ref_252469767172722_1120x1680.png) | ![TensorPencil reproduction](testdata/comfyui_repro_252469767172722_1120x1680.png) |                                        
|                 seed 252469767172722, 20 steps, cfg 1.0                  |                       same settings, 0.57% mean pixel delta                        |


### AI Disclaimer
TensorPencil is heavily AI-assisted code. Most of this stuff is over my head, I'm just tinkering here.
The exception is this readme; I'm of the opinion that if you expect a human to take the time to read something, you should take the time to write it.

## Details

Backends supported so far:
- CPU
- Vulkan
- Zig PTX (CUDA)

The goal is for 100% Zig code other than needed 3rd party libraries for Vulkan and CUDA.

Vulkan so far takes ~1.3x as long as ComfyUI CUDA (warm cache, reference image: ~80 seconds - 3.86s/it in ComfyUI, ~103 seconds / 4.66s/it in TensorPencil). One
limit right now is that there was an issue running flash attention in Vulkan because NVIDIA's 3090 Vulkan driver can only use 48 KB of workgroup shared memory,
whereas CUDA has an opt-in to use 99 KB. This 48KB limit with the current tiling size causes extra traffic when using Flash Attention,
leading to it being *slower* instead of faster in Vulkan. Tried multiple ways to squeeze it in and so far hasn't worked out.

Now also supports INT8 ConvRot.
The speed for this one is farther behind ComfyUI, but roughly matches ComfyUI FP8.
ComfyUI INT8: ~2s/it, TensorPencil INT8: ~4s/it

This has been tested only on Linux with an RTX 3090. It's likely that other operating systems and GPUs will hit problems
or run less efficiently.

Plans for the future:
- Support CUDA in 2 ways for fun:
  - hand-rolled zig PTX (completed)
  - calling out to cuBLASLt directly

## Running it

Requires Zig 0.16.0. There are no build-time C dependencies; the only runtime library is the
Vulkan loader (`libvulkan.so.1`, opened dynamically), and only for `--backend vulkan` — the CPU path
needs nothing. You'll also need a Vulkan driver for your GPU. On Ubuntu:

```
sudo apt install libvulkan1
```

plus a driver: the NVIDIA proprietary driver (e.g. `sudo apt install nvidia-driver-580`)
already includes its Vulkan ICD.

Build with optimizations on (Debug is painfully slow for numeric code):

```
zig build -Doptimize=ReleaseFast
```

Model weights are not included. Place the three Krea 2 checkpoints at these exact paths
(relative to where you run the binary — the paths are currently hardcoded):

```
models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors (or any fp8/int8 krea 2 checkpoint)
models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors (or any qwen 3 VL encoder)
models/vae/krea2RealVae_v10.safetensors (or any WAN2.2-VAE)
```

The Qwen 3 VL tokenizer is embedded in the binary, so no other files are needed. Then:

```
zig-out/bin/TensorPencil generate --prompt "a fluffy orange cat sitting on a windowsill" --backend vulkan --out cat.png
```

Run with no command to see the available options and defaults.

### VRAM offloading

When running on GPU, if other processes are using vram or the `--vram-budget` option is set,
weights past the available budget are streamed from the mmapped file. Assuming sufficient RAM for cache,
this streaming costs only ~20% per step and stays roughly flat across cap sizes — below full residency
effectively every weight re-uploads each step, so a smaller cap is barely slower (see the chart below).
You can pass `min` as the budget size to load only 2 weights at a time, ~150MiB (~40% performance loss per step).

** Note that the VRAM budget is only for the weights, the scores and activations are still in VRAM.**
The amount of VRAM used for the scores and activations depends on the size of the image; at ~1.8MP, it will be roughly 3.1GiB.

Measured on an RTX 3090 at 1120×1680, 4 steps, INT8 ConvRot, vulkan backend:

| VRAM cap                    | s/step | total  |
|:----------------------------|:-------|:-------|
| 0 (driver-managed, default) | 5.25   | 26.4 s |
| 16 GiB                      | 6.39   | 31.0 s |
| 12 GiB                      | 6.45   | 31.1 s |
| 8 GiB                       | 6.46   | 31.1 s |
| 6 GiB                       | 6.42   | 30.9 s |
| 4 GiB                       | 6.33   | 30.5 s |
| 2 GiB                       | 6.47   | 31.2 s |
| 1 GiB                       | 6.53   | 31.3 s |
| min (150MiB)                | 7.21   | 34.2 s |
