"""Time ComfyUI's VAE decode alone, warm, for a speed comparison with the engine's
`generate --decode-bench N`.

Only the VAE is loaded, so this needs a few GB of RAM rather than a whole
pipeline. The latent is random: decode time does not depend on its values.

    /home/qt/genai/comfyui/nvenv/bin/python tools/bench_comfy_vae.py \\
        --vae /mnt/watch/models/diffusion/vae/ae.safetensors --width 1024 --height 1024
"""

import argparse
import os
import sys
import time

COMFY = "/home/qt/genai/comfyui"

ap = argparse.ArgumentParser()
ap.add_argument("--vae", required=True)
ap.add_argument("--width", type=int, default=1024)
ap.add_argument("--height", type=int, default=1024)
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()

sys.argv = [sys.argv[0], "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402

vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))
ds = vae.downscale_ratio
if callable(ds) or isinstance(ds, (tuple, list)):
    # Video VAEs give (temporal, spatial, spatial); a still is one latent frame.
    ds = ds[-1] if isinstance(ds, (tuple, list)) else 8
h, w = args.height // ds, args.width // ds
shape = (1, vae.latent_channels, h, w) if vae.latent_dim == 2 else (1, vae.latent_channels, 1, h, w)
z = torch.randn(shape, generator=torch.Generator().manual_seed(0))
print("vae dtype %s, latent %s" % (vae.vae_dtype, tuple(shape)))

# ComfyUI retries a decode that runs out of memory as a tiled one; count those apart.
tiled = []
_tiled_impl = vae.decode_tiled_


def _tiled(*a, **k):
    tiled[-1] = True
    return _tiled_impl(*a, **k)


vae.decode_tiled_ = _tiled
ms = []
for r in range(args.reps):
    tiled.append(False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = vae.decode(z)
    torch.cuda.synchronize()
    ms.append((time.perf_counter() - t0) * 1e3)
    print("decode %d: %.1f ms%s" % (r + 1, ms[-1], "  (tiled after OOM)" if tiled[-1] else ""))
warm = sorted(ms[1:]) or ms
whole = sorted(m for m, t in zip(ms[1:], tiled[1:]) if not t)
print("%dx%d: first %.1f ms, warm median %.1f ms over %d; whole-image median %s over %d, %d tiled" % (
    args.width, args.height, ms[0], warm[len(warm) // 2], len(warm),
    "%.1f ms" % whole[len(whole) // 2] if whole else "-", len(whole), sum(tiled)))
