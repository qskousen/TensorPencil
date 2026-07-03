## TensorPencil

**This is an experimental work in progress.**

TensorPencil is a test to see how far we can push diffusion performance in pure Zig (aside from Vulkan / CUDA libraries).

It currently targets FP8 Krea 2, and that is the only model that has been tested.
Currently the results differ slightly from ComfyUI but are pretty close - the Zig noise kernel was created to generate bit-identical initial noise,
and any remaining difference is due to the Zig kernels reducing in a different order than cuBLAS/FlashAttention;
over 20 steps that difference accumulates as slight texture-level drift.

|                                 ComfyUI                                  |                                    TensorPencil                                    |                                                                                                                                                                               
|:------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|
| ![ComfyUI reference](testdata/comfyui_ref_252469767172722_1120x1680.png) | ![TensorPencil reproduction](testdata/comfyui_repro_252469767172722_1120x1680.png) |                                        
|                 seed 252469767172722, 20 steps, cfg 1.0                  |                       same settings, 0.61% mean pixel delta                        |


### AI Disclaimer
TensorPencil is heavily AI-assisted code. Most of this stuff is over my head, I'm just tinkering here.
The exception is this readme; I'm of the opinion that if you expect a human to take the time to read something, you should take the time to write it.

Backends supported so far:
- CPU
- Vulkan

The goal is for 100% Zig code other than needed 3rd party libraries for Vulkan and CUDA.

Vulkan so far takes ~1.27x as long as ComfyUI CUDA (reference image: ~85 seconds in ComfyUI, ~108 seconds in TensorPencil). One
limit right now is that there was an issue running flash attention in Vulkan because NVIDIA's 3090 Vulkan driver can only use 48 KB of workgroup shared memory,
whereas CUDA has an opt-in to use 99 KB. This 48KB limit with the current tiling size causes extra traffic when using Flash Attention,
leading to it being *slower* instead of faster in Vulkan. Tried multiple ways to squeeze it in and so far hasn't worked out.

This has been tested only on Linux with an RTX 3090. It's likely that other operating systems and GPUs will hit problems
or run less efficiently.

Plans for the future:
- Support CUDA
- Support 8INT CONVROT format (should allow for speed ~doubling on a 3090)

## Running it

Requires Zig 0.16.0. Build with optimizations on (Debug is painfully slow for numeric code):

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

The tokenizer is embedded in the binary, so no other files are needed. Then:

```
zig-out/bin/TensorPencil generate --prompt "a fluffy orange cat sitting on a windowsill" --gpu on --out cat.png
```

Defaults are 1024x1024 (sizes must be multiples of 16), 8 steps, cfg 1.0, seed 0, shift 1.15 —
the Krea 2 turbo settings. `--gpu on` runs the DiT and VAE decode on Vulkan; the default is the
CPU path, which produces the same image, just much slower. Other flags: `--width`, `--height`,
`--steps`, `--cfg`, `--seed`, `--negative` (needs `--cfg` above 1.0 to have an effect),
`--shift`, `--profile on` (per-step timing breakdown), `--out`.

Expect roughly 14 GB of system RAM at peak (the models load in stages: text encoder, then DiT,
then VAE) and ~18 GB of VRAM for the GPU path at 1 megapixel (measured: 17.5 GiB — the fp8 DiT
weights stay resident and the VAE decode's f32 activations stack on top). A ComfyUI seed
reproduces the same image, so comparisons like the one above are easy to make yourself.