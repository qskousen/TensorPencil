#!/usr/bin/env python3
"""Pin core/resample.zig to Pillow's LANCZOS resize, which ComfyUI's
`common_upscale(..., "lanczos")` runs on every picture it resizes.

Cases cover downscaling (support widened by the scale), upscaling, one axis kept
(that pass is skipped and must not round), and a ratio that is not a fraction of
small integers. Emits src/core/assets/resample_lanczos.bin: per case a u32
header (h, w, out_h, out_w), the source RGB bytes, then Pillow's output.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_resample_fixture.py
"""

import os
import struct

import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "core", "assets", "resample_lanczos.bin")

CASES = [  # (h, w, out_h, out_w)
    (24, 40, 9, 13),
    (9, 13, 31, 22),
    (17, 29, 17, 11),
    (17, 29, 40, 29),
    (33, 21, 20, 50),
]


def main():
    rng = np.random.default_rng(20261110)
    blob = b""
    for h, w, oh, ow in CASES:
        # smooth structure plus noise, so both ringing and clamping occur
        y, x = np.mgrid[0:h, 0:w]
        base = 127 + 120 * np.sin(x[..., None] * 0.7 + y[..., None] * 0.3 + np.arange(3))
        src = np.clip(base + rng.normal(0, 40, (h, w, 3)), 0, 255).astype(np.uint8)
        out = np.asarray(Image.fromarray(src).resize((ow, oh), resample=Image.Resampling.LANCZOS))
        assert out.shape == (oh, ow, 3)
        blob += struct.pack("<4I", h, w, oh, ow) + src.tobytes() + out.tobytes()
    with open(OUT, "wb") as f:
        f.write(blob)
    print("wrote %s: %d cases, %d bytes" % (OUT, len(CASES), len(blob)))


if __name__ == "__main__":
    main()
