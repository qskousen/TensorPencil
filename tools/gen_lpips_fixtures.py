"""Generate the LPIPS reference weights + golden fixtures for src/models/lpips.zig.

Build-time only: this is the *reference* half of the validation, per the project
rule that data-shaped code is validated against an external implementation. The
Zig side never calls Python.

Run with an env that has torch + torchvision + the `lpips` package:

  /home/qt/genai/ai-toolkit/venv/bin/python tools/gen_lpips_fixtures.py

Emits (one fixture file per module that is being validated):
  models/lpips/lpips_alex.safetensors        the weights (NOT committed, ~10 MB)
  src/ops/assets/conv_fixtures.json          torch conv2d / max_pool2d goldens
  src/core/assets/image_metric_fixtures.json skimage SSIM + PSNR/MSE goldens
  src/models/assets/lpips_fixtures.json      LPIPS distances + feature checksums

Three independent validations, deliberately separate:

  1. `conv_fixtures` - hand-sized conv / maxpool cases with the weights inlined,
     so `ops/conv.zig` has a fast unit test that needs no checkpoint. The shapes
     are exactly the ones LPIPS-AlexNet uses, which is why they live here.
  2. `image_metric_fixtures` - small raw-pixel image pairs with skimage SSIM and
     PSNR/MSE, for `core/image.zig`. Pixels inline (small) rather than PNG, so
     the metric test does not depend on the PNG decoder.
  3. `lpips_fixtures` - LPIPS distances for larger PNG-embedded pairs (the Zig
     test decodes the same bytes rather than reproducing an RNG), *plus*
     per-slice feature checksums so a mismatch localizes to a layer instead of
     being "the number is wrong".
"""

import base64
import hashlib
import io as _io
import json
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image
from safetensors.torch import save_file
from skimage.metrics import structural_similarity

import lpips as lpips_pkg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(REPO, "models", "lpips", "lpips_alex.safetensors")
FIXTURES = os.path.join(REPO, "src", "models", "assets", "lpips_fixtures.json")
CONV_FIXTURES = os.path.join(REPO, "src", "ops", "assets", "conv_fixtures.json")
METRIC_FIXTURES = os.path.join(REPO, "src", "core", "assets", "image_metric_fixtures.json")

# torchvision alexnet.features indices that hold the 5 convs LPIPS taps.
CONV_IDX = [0, 3, 6, 8, 10]
CHNS = [64, 192, 384, 256, 256]


# --- images ----------------------------------------------------------------
# Procedural, not photographic: a perceptual metric fed pure noise saturates
# every layer the same way, so the set mixes structure (edges, texture, text-
# like high frequency) with noise, and includes a non-square pair to catch an
# H/W transposition.


def lcg(n, seed):
    """Portable 32-bit LCG (glibc constants), so the recipe is auditable."""
    out = np.empty(n, dtype=np.uint32)
    s = np.uint32(seed)
    for i in range(n):
        s = np.uint32((np.uint64(s) * np.uint64(1103515245) + np.uint64(12345)) & np.uint64(0xFFFFFFFF))
        out[i] = s
    return out


def img_noise(w, h, seed):
    v = (lcg(w * h * 3, seed) >> np.uint32(16)) & np.uint32(0xFF)
    return v.astype(np.uint8).reshape(h, w, 3)


def img_gradient(w, h, phase):
    y, x = np.mgrid[0:h, 0:w]
    r = (x * 255 // max(1, w - 1)).astype(np.int32)
    g = (y * 255 // max(1, h - 1)).astype(np.int32)
    b = ((x + y + phase) * 255 // max(1, w + h - 2)).astype(np.int32)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def img_checker(w, h, cell, shift):
    y, x = np.mgrid[0:h, 0:w]
    a = (((x + shift) // cell) + (y // cell)) % 2
    base = np.where(a == 0, 230, 25).astype(np.int32)
    fine = (((x * 3 + y * 5) % 7) * 3).astype(np.int32)
    return np.clip(np.stack([base, base - fine, base + fine], -1), 0, 255).astype(np.uint8)


def img_rings(w, h, k):
    y, x = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    v = (np.sin(r * (0.6 + 0.02 * k)) * 110 + 128).astype(np.int32)
    u = (np.cos(r * 0.25 + k) * 90 + 128).astype(np.int32)
    return np.clip(np.stack([v, u, (v + u) // 2], -1), 0, 255).astype(np.uint8)


def png_b64(arr):
    buf = _io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_pairs():
    pairs = []
    # 1. two independent noise fields: the metric's upper end.
    pairs.append(("noise64", img_noise(64, 64, 1), img_noise(64, 64, 2)))
    # 2. same structure, small perturbation: the low end that has to stay small.
    a = img_checker(64, 64, 7, 0)
    b = img_checker(64, 64, 7, 1)
    pairs.append(("checker_shift", a, b))
    # 3. structure vs structure+noise: the "damage" case quantization produces.
    base = img_rings(64, 64, 0)
    noisy = np.clip(base.astype(np.int32) + (img_noise(64, 64, 7).astype(np.int32) - 128) // 8, 0, 255).astype(np.uint8)
    pairs.append(("rings_plus_noise", base, noisy))
    # 4. non-square, and a genuinely different picture: catches H/W swaps.
    pairs.append(("grad_vs_rings_96x64", img_gradient(96, 64, 0), img_rings(96, 64, 3)))
    # 5. identical inputs: LPIPS must be exactly 0.
    ident = img_gradient(64, 64, 5)
    pairs.append(("identical", ident, ident.copy()))
    return pairs


# --- weights ---------------------------------------------------------------


def export_weights(model):
    tensors = {}
    feats = model.net
    for n, idx in enumerate(CONV_IDX):
        convs = [m for m in getattr(feats, f"slice{n + 1}") if isinstance(m, torch.nn.Conv2d)]
        assert len(convs) == 1, (n, convs)
        conv = convs[0]
        assert conv.out_channels == CHNS[n], (n, conv)
        tensors[f"features.{idx}.weight"] = conv.weight.detach().contiguous().float()
        tensors[f"features.{idx}.bias"] = conv.bias.detach().contiguous().float()
    for n in range(5):
        lin = getattr(model, f"lin{n}").model[1]
        w = lin.weight.detach().contiguous().float()  # [1, chn, 1, 1]
        assert w.shape == (1, CHNS[n], 1, 1), w.shape
        tensors[f"lin.{n}.weight"] = w.reshape(CHNS[n]).contiguous()
    tensors["scaling.shift"] = model.scaling_layer.shift.detach().reshape(3).contiguous().float()
    tensors["scaling.scale"] = model.scaling_layer.scale.detach().reshape(3).contiguous().float()

    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    save_file(
        tensors,
        WEIGHTS,
        metadata={
            "net": "alex",
            "lpips_version": "0.1",
            "source": "torchvision alexnet IMAGENET1K_V1 features + lpips v0.1 alex lin",
            "generator": "tools/gen_lpips_fixtures.py",
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    )
    for k, v in tensors.items():
        print(f"  {k:24s} {tuple(v.shape)}")
    return tensors


# --- conv / pool goldens ---------------------------------------------------


def ramp(n, seed=3):
    """Deterministic small values in [-1, 1], no RNG dependency either side."""
    i = np.arange(n, dtype=np.float64)
    return (np.sin(i * 0.7 + seed) * 0.5 + np.cos(i * 0.13) * 0.5).astype(np.float32)


def conv_case(name, h, w, ci, co, k, stride, pad):
    x = ramp(h * w * ci, 1).reshape(1, ci, h, w)  # [1,ci,h,w] channel-first
    wt = ramp(co * ci * k * k, 2).reshape(co, ci, k, k)
    b = ramp(co, 5)
    y = torch.nn.functional.conv2d(
        torch.from_numpy(x), torch.from_numpy(wt), torch.from_numpy(b), stride=stride, padding=pad
    ).numpy()
    oh, ow = y.shape[2], y.shape[3]
    # Emit channel-LAST, which is the layout tp's conv works in.
    return {
        "name": name,
        "h": h, "w": w, "ci": ci, "co": co, "k": k, "stride": stride, "pad": pad,
        "out_h": oh, "out_w": ow,
        "x": [float(v) for v in x.transpose(0, 2, 3, 1).reshape(-1)],
        "weight": [float(v) for v in wt.reshape(-1)],  # [co][ci][k][k], as torch stores it
        "bias": [float(v) for v in b.reshape(-1)],
        "out": [float(v) for v in y.transpose(0, 2, 3, 1).reshape(-1)],
    }


def pool_case(name, h, w, c, k, stride):
    x = ramp(h * w * c, 4).reshape(1, c, h, w)
    y = torch.nn.functional.max_pool2d(torch.from_numpy(x), kernel_size=k, stride=stride).numpy()
    return {
        "name": name, "h": h, "w": w, "c": c, "k": k, "stride": stride,
        "out_h": y.shape[2], "out_w": y.shape[3],
        "x": [float(v) for v in x.transpose(0, 2, 3, 1).reshape(-1)],
        "out": [float(v) for v in y.transpose(0, 2, 3, 1).reshape(-1)],
    }


# --- main ------------------------------------------------------------------


def metric_goldens(a, b):
    """The four figures core/image.zig owes an external reference."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    psnr = None if mse == 0 else 10.0 * np.log10(255.0 * 255.0 / mse)
    return {
        "mse": mse,
        # null means +inf (identical inputs); JSON has no infinity.
        "psnr": psnr,
        # skimage defaults: 7x7 uniform window, sample covariance, per-channel
        # then averaged.
        "ssim": float(structural_similarity(a, b, channel_axis=-1, data_range=255)),
        # the Wang et al. convention the SSIM paper actually specifies.
        "ssim_gaussian": float(
            structural_similarity(a, b, channel_axis=-1, data_range=255, gaussian_weights=True,
                                  sigma=1.5, use_sample_covariance=False)
        ),
    }


def write_json(path, obj, label):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        # compact: the arrays are machine-read, and one element per line triples
        # the file for no benefit.
        json.dump(obj, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    print(f"{label} -> {path} ({os.path.getsize(path)} bytes)")


def versions():
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "lpips": getattr(lpips_pkg, "__version__", "0.1.4"),
        "skimage": __import__("skimage").__version__,
        "python": sys.version.split()[0],
    }


def main():
    header = {"generated_by": "tools/gen_lpips_fixtures.py", "versions": versions()}

    # --- 1. ops/conv.zig -------------------------------------------------
    write_json(CONV_FIXTURES, dict(
        header,
        conv_cases=[
            # the three shapes LPIPS-alexnet uses, plus a 1x1 and a stride that
            # does not divide the input (floor semantics).
            conv_case("k11s4p2", 24, 20, 3, 5, 11, 4, 2),
            conv_case("k5s1p2", 11, 9, 4, 6, 5, 1, 2),
            conv_case("k3s1p1", 7, 6, 3, 4, 3, 1, 1),
            conv_case("k1s1p0", 5, 4, 3, 2, 1, 1, 0),
            conv_case("k3s2p0", 8, 7, 2, 3, 3, 2, 0),
        ],
        pool_cases=[
            pool_case("k3s2", 15, 13, 3, 3, 2),
            pool_case("k3s2_even", 8, 8, 2, 3, 2),
        ],
    ), "conv fixtures")

    # --- 2. core/image.zig metrics ---------------------------------------
    metric_pairs = []
    for name, a, b in [
        ("noise16", img_noise(16, 16, 11), img_noise(16, 16, 12)),
        ("checker16", img_checker(16, 16, 4, 0), img_checker(16, 16, 4, 1)),
        ("rings24x16", img_rings(24, 16, 0), img_rings(24, 16, 1)),
        ("gradient_vs_self", img_gradient(16, 16, 0), img_gradient(16, 16, 0)),
        # 12x12 is the smallest that admits the gaussian variant's 11x11 window,
        # so it is where a window-placement bug shows up.
        ("small12", img_gradient(12, 12, 0), img_checker(12, 12, 3, 0)),
    ]:
        h, w = a.shape[0], a.shape[1]
        metric_pairs.append(dict(
            {"name": name, "width": w, "height": h,
             "a": [int(v) for v in a.reshape(-1)], "b": [int(v) for v in b.reshape(-1)]},
            **metric_goldens(a, b),
        ))
        print(f"  {name:22s} {w}x{h}  ssim={metric_pairs[-1]['ssim']:.6f}")
    write_json(METRIC_FIXTURES, dict(header, pairs=metric_pairs), "metric fixtures")

    # --- 3. models/lpips.zig ---------------------------------------------
    model = lpips_pkg.LPIPS(net="alex", version="0.1")
    model.eval()

    print("weights ->", WEIGHTS)
    export_weights(model)
    with open(WEIGHTS, "rb") as f:
        wsha = hashlib.sha256(f.read()).hexdigest()

    pairs = []
    for name, a, b in make_pairs():
        h, w = a.shape[0], a.shape[1]
        assert b.shape == a.shape
        ta = torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)[None] * 2 - 1
        tb = torch.from_numpy(b.astype(np.float32) / 255.0).permute(2, 0, 1)[None] * 2 - 1
        with torch.no_grad():
            val, per_layer = model(ta, tb, retPerLayer=True)
            # per-slice feature checksums, on the *pre*-normalization features
            # (what conv+relu produced) so a mismatch localizes to a conv.
            outs_a = model.net.forward(model.scaling_layer(ta))
            outs_b = model.net.forward(model.scaling_layer(tb))

        def sums(outs):
            r = []
            for t in outs:
                f = t.detach().double()
                r.append({
                    "c": int(t.shape[1]), "h": int(t.shape[2]), "w": int(t.shape[3]),
                    "sum": float(f.sum()), "sumsq": float((f * f).sum()), "max": float(f.max()),
                })
            return r

        pairs.append({
            "name": name, "width": w, "height": h,
            "png_a": png_b64(a), "png_b": png_b64(b),
            "lpips": float(val.reshape(-1)[0]),
            "per_layer": [float(t.reshape(-1)[0]) for t in per_layer],
            "feat_a": sums(outs_a), "feat_b": sums(outs_b),
        })
        print(f"  {name:22s} {w}x{h}  lpips={float(val.reshape(-1)[0]):.8f}")

    write_json(FIXTURES, dict(
        header,
        net="alex",
        lpips_version="0.1",
        weights_file="models/lpips/lpips_alex.safetensors",
        weights_sha256=wsha,
        chns=CHNS,
        pairs=pairs,
    ), "lpips fixtures")


if __name__ == "__main__":
    main()
