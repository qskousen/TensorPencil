#!/usr/bin/env python3
"""Reference fixtures for ComfyUI's NVFP4 weight format.

Every number comes from `comfy_kitchen`'s own implementation
(`backends.eager.quantization.dequantize_nvfp4` and `float_utils.to_blocked`), never
from ours. A layer ships:

    weight          U8   [rows, cols/2]   two E2M1 codes per byte
    weight_scale    F8_E4M3               per-16-element block scale, SWIZZLED
    weight_scale_2  F32   scalar          per-tensor scale
    input_scale     F32   scalar          static ACTIVATION scale, Blackwell-only

and decodes as

    total[row, blk] = weight_scale_2 * fp8(block_scale[row, blk])
    value           = E2M1[nibble] * total[row, blk]

⚠️ Three conventions that a port would get wrong by analogy with the int4/W4A8 formats
already in this engine, so each gets a case whose *teeth are asserted below*:

  1. **`hi_first = True`**: element 2k is the HIGH nibble. `.i4` convrot and W4A8 here
     are both low-nibble-first. Getting it backwards swaps every adjacent weight pair —
     rms-preserving, so no magnitude check sees it.
  2. **The block scales are SWIZZLED** into cuBLAS's tiled layout (`to_blocked`:
     `(32*ceil(rows,128), 16*ceil(cols/16,4))` element order). On disk they carry the
     LOGICAL `[rows, cols/16]` *shape*, so nothing about the header hints at it; only
     the order differs. W4A8's `s_rel` is plain row-major, which is exactly the wrong
     prior to bring here.
  3. **The multiply ASSOCIATION is `E2M1 * (per_tensor * block)`**, not
     `(E2M1 * block) * per_tensor`. Only the first is bit-exact against the reference,
     and it is the one a per-scale-byte lookup table computes naturally.

Reference output dtype is **f32** here on purpose. ComfyUI itself dequantizes to the
model's `orig_dtype` (bf16 for these checkpoints), which would put a ~0.4% floor in the
*reference*; f32 makes the fixture exact, and the bf16 floor is then a property of
ComfyUI's render rather than of this comparison.

Two tiers: synthetic (`src/ops/assets/nvfp4_fixtures.json`, ungated) and one real layer
per family (`src/models/assets/nvfp4_real_layers.json`, gated, hashed inputs+output).

Usage:
    ~/genai/comfyui/nvenv/bin/python tools/gen_nvfp4_fixtures.py
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

from comfy_kitchen.backends.eager.quantization import E2M1_LUT, dequantize_nvfp4
from comfy_kitchen.float_utils import from_blocked, to_blocked

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "ops", "assets", "nvfp4_fixtures.json")
OUT_REAL = os.path.join(
    os.path.dirname(__file__), "..", "src", "models", "assets", "nvfp4_real_layers.json"
)
MODELS = os.path.expanduser("~/genai/comfyui/models/")
REAL = [
    ("krea2", "diffusion_models/krea2/krea2CenterSemiraw_v10Int8AndBf16-NVFP4.safetensors",
     "model.diffusion_model.blocks.0.attn.wk"),
    ("anima", "diffusion_models/anima/terraRising_20TerraRisingAnima-NVFP4.safetensors", None),
    ("zimage", "checkpoints/zit/zImageTurboNvfp4_v10.safetensors", None),
]


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def fnv_np(a) -> int:
    return fnv1a64(np.ascontiguousarray(a).tobytes())


def make_case(name, rows, cols, seed):
    """One synthetic case, decoded by the reference."""
    assert cols % 16 == 0
    rng = np.random.default_rng(seed)
    nblk = cols // 16

    # Packed E2M1: random nibbles, both signs and every magnitude.
    qdata = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)

    # Logical per-block scales, then SWIZZLED exactly as a checkpoint stores them.
    # Span several fp8 binades so the exponent range is exercised; 1.25 is exact in
    # e4m3 so each stored byte is unambiguous.
    logical = (2.0 ** rng.integers(-4, 5, size=(rows, nblk))).astype(np.float32) * np.float32(1.25)
    logical_t = torch.from_numpy(logical).to(torch.float8_e4m3fn)
    blocked = to_blocked(logical_t.view(torch.uint8).view(rows, nblk))
    scale_bytes = blocked.contiguous().view(torch.uint8).numpy().tobytes()
    assert len(scale_bytes) >= rows * nblk

    global_scale = np.float32(0.03125 * 1.5)  # exact in f32 and f8, no rounding games

    got = dequantize_nvfp4(
        torch.from_numpy(qdata),
        torch.tensor(global_scale, dtype=torch.float32),
        blocked.view(torch.float8_e4m3fn),
        output_type=torch.float32,
    ).numpy()
    assert got.shape == (rows, cols), got.shape

    return {
        "name": name,
        "rows": rows,
        "cols": cols,
        "global_scale": float(global_scale),
        "packed_hex": qdata.tobytes().hex(),
        # The bytes as stored: swizzled, and only the first rows*nblk of them are the
        # ones a decode needs (the layout pads to 128x4 blocks).
        "scale_blocked_hex": scale_bytes.hex(),
        "scale_logical_hex": logical_t.view(torch.uint8).numpy().tobytes().hex(),
        "expect_f32": [float(v) for v in got.reshape(-1)],
    }


def synthetic_cases():
    cases = []
    # A shape whose scale grid needs NO swizzle padding (rows % 128 == 0, nblk % 4 == 0),
    # so the swizzle is a pure permutation — the case that isolates convention 2.
    cases.append(make_case("aligned_128x64", 128, 64, seed=1))
    # Padded on both axes: rows not a multiple of 128 and nblk not a multiple of 4, so
    # `from_blocked` has to slice a padded grid back down.
    cases.append(make_case("padded_40x48", 40, 48, seed=2))
    # Several row blocks AND several column blocks, both padded: 160 rows pads to 2x128
    # and 6 block-columns pad to 2x4, so the swizzle's block structure is exercised in
    # both axes at once rather than one at a time.
    cases.append(make_case("multi_block_160x96", 160, 96, seed=3))
    # A single 16-element block per row: the degenerate grid.
    cases.append(make_case("one_block_16x16", 16, 16, seed=4))
    return cases


def read_header(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return f, h, 8 + n


def tensor_bytes(f, base, h, name):
    o = h[name]["data_offsets"]
    f.seek(base + o[0])
    return f.read(o[1] - o[0]), h[name]["shape"], h[name]["dtype"]


def real_layer(fam, rel, layer):
    path = MODELS + rel
    if not os.path.exists(path):
        print(f"  (skipping {fam}: {rel} not present)", file=sys.stderr)
        return None
    f, h, base = read_header(path)
    if layer is None:
        cands = sorted(k[: -len(".weight_scale_2")] for k in h if k.endswith(".weight_scale_2"))
        layer = cands[0]
    wb, wshape, wdt = tensor_bytes(f, base, h, layer + ".weight")
    sb, sshape, sdt = tensor_bytes(f, base, h, layer + ".weight_scale")
    gb, _, _ = tensor_bytes(f, base, h, layer + ".weight_scale_2")
    f.close()
    assert wdt == "U8" and sdt == "F8_E4M3", (wdt, sdt)

    rows, cols = wshape[0], wshape[1] * 2
    global_scale = np.frombuffer(gb, dtype=np.float32)[0]
    blocked = torch.from_numpy(np.frombuffer(sb, dtype=np.uint8).copy()).view(sshape)
    got = dequantize_nvfp4(
        torch.from_numpy(np.frombuffer(wb, dtype=np.uint8).reshape(rows, cols // 2).copy()),
        torch.tensor(float(global_scale), dtype=torch.float32),
        blocked.view(torch.float8_e4m3fn),
        output_type=torch.float32,
    ).numpy()

    return {
        "family": fam,
        "checkpoint": os.path.basename(rel),
        "layer": layer,
        "rows": rows,
        "cols": cols,
        "global_scale": float(global_scale),
        # Hash the INPUTS as well as the output: if the checkpoint changes, that is a
        # different fact from "the decode is wrong", and only the input hash can say so.
        "packed_fnv1a64": fnv_np(np.frombuffer(wb, dtype=np.uint8)),
        "scale_fnv1a64": fnv_np(np.frombuffer(sb, dtype=np.uint8)),
        "expect_f32_fnv1a64": fnv_np(got.astype(np.float32)),
        "expect_head": [float(v) for v in got.reshape(-1)[:32]],
        "expect_tail": [float(v) for v in got.reshape(-1)[-32:]],
    }


def check_teeth(cases):
    """Each documented convention must actually change the answer on some case."""
    lut = E2M1_LUT.flatten().numpy().astype(np.float32)
    c = next(x for x in cases if x["name"] == "aligned_128x64")
    rows, cols = c["rows"], c["cols"]
    nblk = cols // 16
    q = np.frombuffer(bytes.fromhex(c["packed_hex"]), dtype=np.uint8).reshape(rows, cols // 2)
    want = np.array(c["expect_f32"], dtype=np.float32).reshape(rows, cols)
    blocked = torch.from_numpy(
        np.frombuffer(bytes.fromhex(c["scale_blocked_hex"]), dtype=np.uint8).copy()
    )
    logical = from_blocked(blocked.view(rows, nblk), rows, nblk).view(torch.float8_e4m3fn).float().numpy()
    g = np.float32(c["global_scale"])

    def decode(hi_first, scales):
        hi = (q >> 4).astype(np.int32)
        lo = (q & 0xF).astype(np.int32)
        codes = np.empty((rows, cols), dtype=np.int32)
        if hi_first:
            codes[:, 0::2], codes[:, 1::2] = hi, lo
        else:
            codes[:, 0::2], codes[:, 1::2] = lo, hi
        total = (g * scales).astype(np.float32)
        return (lut[codes].reshape(rows, nblk, 16) * total[:, :, None]).reshape(rows, cols)

    assert np.array_equal(decode(True, logical), want), "reference disagrees with our model of it"
    n_lo = int((decode(False, logical) != want).sum())
    # The swizzled bytes read as if they were row-major: what skipping from_blocked does.
    raw = blocked.view(torch.float8_e4m3fn).float().numpy().reshape(rows, nblk)
    n_sw = int((decode(True, raw) != want).sum())
    assert n_lo > 0 and n_sw > 0, f"toothless: lo-first {n_lo}, no-unswizzle {n_sw}"
    print(f"  teeth: low-nibble-first moves {n_lo}/{rows*cols} values, "
          f"skipping the unswizzle moves {n_sw}/{rows*cols}")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    torch.manual_seed(0)

    cases = synthetic_cases()
    check_teeth(cases)
    banner = ("Generated by tools/gen_nvfp4_fixtures.py from comfy_kitchen's own NVFP4 "
              "reference. Do not hand-edit.")
    out = {
        "_comment": banner,
        "e2m1_lut": [float(v) for v in E2M1_LUT.flatten().numpy()],
        "cases": cases,
    }
    path = os.path.normpath(OUT)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path} ({os.path.getsize(path)/1024:.0f} KB, {len(cases)} cases)")

    reals = [r for r in (real_layer(*x) for x in REAL) if r is not None]
    if reals:
        rpath = os.path.normpath(OUT_REAL)
        with open(rpath, "w") as f:
            json.dump({"_comment": banner, "layers": reals}, f, indent=1)
        print(f"wrote {rpath} ({len(reals)} real layers: "
              f"{', '.join(r['family'] for r in reals)})")


if __name__ == "__main__":
    main()
