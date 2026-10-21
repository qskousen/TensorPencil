#!/usr/bin/env python3
"""Reference data for EmbeddingGemma 2's image path, by running transformers'
processor and model and sentence-transformers' pipeline (f32, eager, CPU).

Two outputs:

`src/core/assets/resize_aa_bicubic.safetensors` pins the resize the processor
does: torchvision `resize(uint8, bicubic, antialias=True)`, which is PIL's
filter (cubic a = -0.5, support widened by the downscale factor) with a rounded
uint8 store after the horizontal pass and again after the vertical one. A down
and an up resize of one source, on a non-square shape so a swapped axis fails.

`testdata/embeddinggemma2_image_ref.safetensors`, with the three input images
beside it as PNGs:
  - `vec.<name>` / `ids.<name>`: sentence-transformers' vector and the token ids
    for each image at the default 280-token budget, no prompt (the model card
    says images take none).
  - `small.*`: the `down` image at a 70-token budget, through the model
    directly. `small.pixels` is what the processor resized it to, `small.soft`
    the projected soft tokens, `small.vec` the vector.
    Feeding `small.pixels` to a port isolates its tower from its resize.

The manual model path is checked against sentence-transformers before use.

    systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \\
        /dump/venv-eg2/bin/python tools/gen_embeddinggemma2_image.py
"""

import os
import sys

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer

MODEL = "/dump/projects/zig/DiffKeep/Models/embeddinggemma-2"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTDATA = os.path.join(ROOT, "testdata")
CORE_ASSETS = os.path.join(ROOT, "src", "core", "assets")


def synth(w, h, seed):
    """Smooth color fields plus fine stripes (which only an antialiased resize
    averages away) and a few hard-edged disks."""
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    xn, yn = x / w, y / h
    r = 127 + 120 * np.sin(2 * np.pi * (2.5 * xn + 1.3 * yn) + rng.rand())
    g = 255 * (0.3 + 0.6 * yn) * (0.5 + 0.5 * np.cos(2 * np.pi * xn * 1.7))
    b = 255 * ((x.astype(int) // 3 + y.astype(int) // 5) % 2)
    img = np.stack([r, g, b], -1)
    for _ in range(4):
        cx, cy, rad = rng.rand() * w, rng.rand() * h, (0.08 + 0.15 * rng.rand()) * min(w, h)
        mask = (x - cx) ** 2 + (y - cy) ** 2 < rad ** 2
        img[mask] = rng.randint(0, 256, 3)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def resize_u8(hwc, th, tw):
    t = torch.from_numpy(hwc).permute(2, 0, 1).contiguous()
    out = tvF.resize(t, size=[th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    return out.permute(1, 2, 0).contiguous()


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def core_fixture():
    src = synth(160, 120, 7)
    out = {"src": torch.from_numpy(src)}
    for name, (th, tw) in {"down": (48, 96), "up": (144, 208)}.items():
        out[name] = resize_u8(src, th, tw)
    path = os.path.join(CORE_ASSETS, "resize_aa_bicubic.safetensors")
    save_file(out, path)
    print("wrote", path, os.path.getsize(path) // 1024, "KiB")


def model_fixture():
    st = SentenceTransformer(MODEL, device="cpu", model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"})
    model = st[0].model
    proc = st[0].processor

    def manual(img, budget):
        inputs = proc(images=[img], max_soft_tokens=budget, return_tensors="pt")
        with torch.no_grad():
            h = model(**inputs).last_hidden_state[0]
        v = h.mean(0)
        return inputs, v / v.norm()

    out = {}
    images = {"down": synth(1200, 900, 1), "up": synth(120, 90, 2), "tall": synth(64, 300, 3)}
    for name, arr in images.items():
        Image.fromarray(arr).save(os.path.join(TESTDATA, f"embeddinggemma2_{name}.png"))
        img = Image.fromarray(arr)
        with torch.no_grad():
            v = st.encode([img], convert_to_tensor=True)[0].float()
        inputs, vm = manual(img, 280)
        c = cos(v, vm)
        if c < 0.999999:
            sys.exit(f"{name}: manual path disagrees with sentence-transformers (cos {c})")
        ids = inputs["input_ids"][0]
        out[f"vec.{name}"] = v.contiguous()
        out[f"ids.{name}"] = ids.to(torch.int32).contiguous()
        print(f"{name}: {arr.shape[1]}x{arr.shape[0]} -> {len(ids)} tokens")

    img = Image.fromarray(images["down"])
    inputs, v = manual(img, 70)
    pv = inputs["pixel_values"][0]  # [max_patches, 16*16*3], patch rows in (y, x, c) order
    pos = inputs["image_position_ids"][0]
    valid = (pos >= 0).all(-1)
    gx, gy = int(pos[valid, 0].max()) + 1, int(pos[valid, 1].max()) + 1
    px = pv[valid].reshape(gy, gx, 16, 16, 3).permute(0, 2, 1, 3, 4).reshape(gy * 16, gx * 16, 3)
    pixels = torch.round(px * 255).clamp(0, 255).to(torch.uint8)
    if (pixels.float() / 255 - px).abs().max() > 1e-6:
        sys.exit("processor pixels are not exact uint8 levels")
    with torch.no_grad():
        soft = model.get_image_features(inputs["pixel_values"], inputs["image_position_ids"]).pooler_output[0]
    out["small.pixels"] = pixels.contiguous()
    out["small.soft"] = soft.float().contiguous()
    out["small.vec"] = v.float().contiguous()
    out["small.ids"] = inputs["input_ids"][0].to(torch.int32).contiguous()
    print(f"small: {gx}x{gy} patches -> {soft.shape[0]} soft tokens")

    path = os.path.join(TESTDATA, "embeddinggemma2_image_ref.safetensors")
    save_file(out, path)
    print("wrote", path, os.path.getsize(path) // 1024, "KiB")


if __name__ == "__main__":
    core_fixture()
    model_fixture()
