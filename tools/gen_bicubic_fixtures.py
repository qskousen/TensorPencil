#!/usr/bin/env python3
"""Pin `torch.nn.functional.interpolate(mode="bicubic")` at its defaults, which
is the resize ComfyUI's `common_upscale(..., "bicubic", "disabled")` performs and
the one Mage-Flow-Edit caps a reference image's long edge with.

Anything that merely "resizes bicubically" produces a plausible picture; only
matching torch's kernel reproduces the reference's vision tokens. Three pieces
are silent when wrong:

  - the cubic coefficient A is **-0.75**, not the -0.5 Catmull-Rom uses;
  - `align_corners=False`, so the sample sits at `(i + 0.5) * scale - 0.5`, and
    unlike the BILINEAR path torch does NOT clamp that to zero for cubic -- a
    negative source coordinate is real and its weights are what handle the edge;
  - taps are gathered with the source index CLAMPED to the image, so the border
    replicates rather than going to zero.

Both directions are covered (down is the case Mage-Flow actually takes, up is
where a wrong `align_corners` shows most), plus a non-square shape, because a
transposed axis is exactly the bug a square case cannot see.

    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_bicubic_fixtures.py
"""

import json
import os

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "src", "core", "assets", "bicubic_fixtures.json")

# (name, src h, src w, dst h, dst w)
CASES = [
    ("down", 11, 7, 5, 3),
    ("up", 3, 4, 7, 9),
    ("wide", 5, 13, 4, 6),
    ("same", 6, 6, 6, 6),
]


def main() -> None:
    torch.manual_seed(20260918)
    cases = []
    for name, sh, sw, dh, dw in CASES:
        src = torch.randn(1, 3, sh, sw, dtype=torch.float32)
        dst = torch.nn.functional.interpolate(src, size=(dh, dw), mode="bicubic")
        cases.append({
            "name": name,
            "sh": sh, "sw": sw, "dh": dh, "dw": dw,
            # Planar [3][h][w], the layout the Zig side works in.
            "src": src[0].reshape(-1).tolist(),
            "dst": dst[0].reshape(-1).tolist(),
        })
        print(f"{name}: {sh}x{sw} -> {dh}x{dw}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"torch": torch.__version__, "cases": cases}, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
