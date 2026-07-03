## TensorPencil

**This is an experimental work in progress.**

TensorPencil is a test to see how far we can push diffusion performance in pure Zig (aside from Vulkan / CUDA libraries).

It currently targets FP8 Krea 2, and that is the only model that has been tested.
Currently the results differ slightly from ComfyUI but are pretty close - the Zig noise kernel was created to generate bit-identical initial noise,
and any remaining difference is due to the Zig DiT kernels reducing in a different order than cuBLAS/FlashAttention;
over 20 steps that difference accumulates as slight texture-level drift.

|                                 ComfyUI                                  |                                    TensorPencil                                    |                                                                                                                                                                               
|:------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|
| ![ComfyUI reference](testdata/comfyui_ref_252469767172722_1120x1680.png) | ![TensorPencil reproduction](testdata/comfyui_repro_252469767172722_1120x1680.png) |                                        
|                 seed 252469767172722, 20 steps, cfg 1.0                  |                       same settings, 0.57% mean pixel delta                        |


### AI Disclaimer
TensorPencil is heavily AI-assisted code. Most of this stuff is over my head, I'm just tinkering here.
The exception is this readme; I'm of the opinion that if you expect a human to take the time to read something, you should take the time to write it.

Backends supported so far:
- CPU
- Vulkan

The goal is for 100% Zig code other than needed 3rd party libraries for Vulkan and CUDA.

Vulkan so far takes ~1.3x as long as ComfyUI CUDA (warm cache, reference image: ~80 seconds - 3.86s/it in ComfyUI, ~103 seconds / 4.66s/it in TensorPencil). One
limit right now is that there was an issue running flash attention in Vulkan because NVIDIA's 3090 Vulkan driver can only use 48 KB of workgroup shared memory,
whereas CUDA has an opt-in to use 99 KB. This 48KB limit with the current tiling size causes extra traffic when using Flash Attention,
leading to it being *slower* instead of faster in Vulkan. Tried multiple ways to squeeze it in and so far hasn't worked out.

This has been tested only on Linux with an RTX 3090. It's likely that other operating systems and GPUs will hit problems
or run less efficiently.

Plans for the future:
- Support CUDA
- Support 8INT CONVROT format (should allow for speed ~doubling on a 3090)

## Running it

Requires Zig 0.16.0. There are no build-time C dependencies; the only runtime library is the
Vulkan loader (`libvulkan.so.1`, opened dynamically), and only for `--gpu on` — the CPU path
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
models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors
models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors
models/vae/krea2RealVae_v10.safetensors
```

The Qwen 3 VL tokenizer is embedded in the binary, so no other files are needed. Then:

```
zig-out/bin/TensorPencil generate --prompt "a fluffy orange cat sitting on a windowsill" --gpu on --out cat.png
```

Run with no command to see the available options and defaults.

### VRAM offloading

When running on GPU, if other processes are using vram or the `--vram-budget` option is set,
weights past the available budget are streamed from the mmapped file. Assuming sufficient RAM,
this streaming is nearly free performance-wise, as displayed in the chart below, tested all the
way down to 1 GB of VRAM.

Measured on an RTX 3090 at 1120×1680, 4 steps, f16 DiT:

| VRAM cap                    | s/step | total  |
|:----------------------------| :----- | :----- |
| 0 (driver-managed, default) | 5.03   | 30.0 s |
| 16 GiB                      | 6.04   | 33.9 s |
| 12 GiB                      | 6.06   | 34.1 s |
| 8 GiB                       | 6.23   | 34.7 s |
| 6 GiB                       | 6.00   | 33.7 s |
| 4 GiB                       | 6.05   | 34.0 s |
| 2 GiB                       | 6.03   | 34.0 s |
| 1 GiB                       | 6.04   | 33.8 s |
