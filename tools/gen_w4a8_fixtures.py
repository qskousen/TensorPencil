#!/usr/bin/env python3
"""Reference fixtures for ComfyUI's `asym_w4a8_int8` weight format.

Every number here comes from `comfy_kitchen`'s own implementation
(`comfy_kitchen.backends.eager.w4a8_int8`) — the code ComfyUI actually runs — never
from ours. The format stores a 4-bit weight that *decodes to an int8 convrot weight*:

    q      = nibble(packed)                       # 0..15, even col = LOW nibble
    level  = codebook[q]      (or q - 8 when the layer ships no codebook)
    int8   = rint(clamp(level * s_rel[row, group], -127, 127))

and from there it is the int8-convrot GEMM this engine already implements
(per-output-row `s_channel`, Hadamard-256 activation rotation). So the *only* new
arithmetic is that decode, which is what tier 1 below pins exactly.

⚠️ `rint` is round-half-to-EVEN. torch's `.round()` and the Triton kernel's
`libdevice.rint` both are; Zig's `@round` is half-away-from-zero. The `ties` case
below is built so those two disagree on 8 of 16 levels — without it the difference
is invisible on real weights, where an exact tie is rare but not impossible.

Two tiers, because they answer different questions:

  1. **Ops** (`src/ops/assets/w4a8_fixtures.json`, ~200 KB, ungated). Synthetic
     weights through the reference decode, plus the full `dequantize_w4a8_int8_weight`
     (which un-rotates back to the original basis) so the convrot half is covered
     too. No checkpoint, so it runs in the fast suite anywhere.

  2. **A real layer** (`src/models/assets/w4a8_real_layer.json`, ~1 KB, gated). One
     layer read straight out of the safetensors payload and decoded by the reference;
     we record FNV-1a 64 over the input bytes AND over the expected int8 output.
     Hashing the inputs rather than the whole 8.8 GB file is what ties the expectation
     to the bytes that actually matter: if the checkpoint changes, the input hash moves
     and the test says so instead of comparing against stale weights. This tier exists
     to catch *container* mistakes the synthetic tier cannot — the [N, K/2] shape
     doubling, the fp8 view of `weight_s_rel`, and the `comfy_quant` parse — so it
     lives in the module that owns the loader (`tp_models`) rather than in `tp_ops`,
     which cannot reach `test_gate`.

Usage (needs ComfyUI's own env, which is where comfy_kitchen lives):
    ~/genai/comfyui/nvenv/bin/python tools/gen_w4a8_fixtures.py
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

from comfy_kitchen.backends.eager.w4a8_int8 import (
    _FIXED_LUT,
    _dequant_int4_grouped_to_int8,
    dequantize_w4a8_int8_weight,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "ops", "assets", "w4a8_fixtures.json")
OUT_REAL = os.path.join(
    os.path.dirname(__file__), "..", "src", "models", "assets", "w4a8_real_layer.json"
)
CKPT = os.path.expanduser(
    "~/genai/comfyui/models/diffusion_models/krea2/"
    "krea2CenterSemiraw_v10Int8-ASYM_W4A8_INT8.safetensors"
)
REAL_LAYER = "blocks.0.attn.wk"


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def fnv_np(arr: np.ndarray) -> int:
    """FNV-1a over the raw bytes; np.uint64 loop in C would be nicer but this is fine."""
    return fnv1a64(np.ascontiguousarray(arr).tobytes())


def pack_nibbles(q: np.ndarray) -> np.ndarray:
    """[n, k] uint4 values -> [n, k/2] int8, element 2i in the low nibble."""
    lo = q[:, 0::2] & 0xF
    hi = q[:, 1::2] & 0xF
    return ((lo | (hi << 4)).astype(np.uint8)).view(np.int8)


def make_case(name, rows, cols, group_size, codebook, s_rel_f32, seed, convrot_groupsize=256):
    """One synthetic case, decoded by the reference."""
    rng = np.random.default_rng(seed)
    groups = cols // group_size
    q = rng.integers(0, 16, size=(rows, cols), dtype=np.uint8)
    packed = pack_nibbles(q)

    # Round-trip the group scale through fp8 e4m3 so the fixture stores exactly what a
    # checkpoint stores and the Zig side decodes the same bytes.
    s_rel_t = torch.from_numpy(s_rel_f32).to(torch.float8_e4m3fn)
    s_rel_bytes = s_rel_t.view(torch.uint8).numpy().tobytes()
    assert len(s_rel_bytes) == rows * groups

    cb_t = None if codebook is None else torch.tensor(codebook, dtype=torch.float32)
    qdata = torch.from_numpy(packed)

    got_i8 = _dequant_int4_grouped_to_int8(qdata, s_rel_t, cb_t, group_size)

    # s_channel is only used by the second half (channel scale + un-rotation); pick
    # something non-uniform so a per-row bug cannot hide behind a constant.
    s_channel = (0.5 + np.arange(rows, dtype=np.float32) * 0.125) / 64.0
    deq = dequantize_w4a8_int8_weight(
        qdata,
        s_rel_t,
        torch.from_numpy(s_channel),
        codebook=cb_t,
        correction=None,
        group_size=group_size,
        convrot_groupsize=convrot_groupsize,
        output_dtype=torch.float32,
    )

    return {
        "name": name,
        "rows": rows,
        "cols": cols,
        "group_size": group_size,
        "convrot_groupsize": convrot_groupsize,
        "codebook": None if codebook is None else [float(v) for v in codebook],
        "packed_hex": packed.tobytes().hex(),
        "s_rel_hex": s_rel_bytes.hex(),
        "s_channel": [float(v) for v in s_channel],
        "expect_i8": [int(v) for v in got_i8.numpy().reshape(-1)],
        "expect_deq_f32": [float(v) for v in deq.numpy().reshape(-1)],
    }


def synthetic_cases():
    cases = []
    lut = list(_FIXED_LUT)

    # 1. The ordinary shape: the frozen Lloyd-Max codebook, group_size 16, scales
    #    spanning a few binades so the fp8 exponent range is exercised.
    rng = np.random.default_rng(1)
    rows, cols, gs = 8, 256, 16
    s = (2.0 ** rng.integers(-3, 6, size=(rows, cols // gs))).astype(np.float32)
    s *= np.float32(1.25)  # 1.25 is exact in e4m3, so the stored byte is unambiguous
    cases.append(make_case("fixed_lut_gs16", rows, cols, gs, lut, s, seed=11))

    # 2. No codebook: levels are the uniform q-8. A layer quantized with
    #    `codebook=False` ships no `weight_codebook`, and the two branches differ in
    #    every value, so a reader that ignores the tensor's absence fails here.
    rows, cols, gs = 4, 512, 16
    s = np.full((rows, cols // gs), 4.0, dtype=np.float32)
    cases.append(make_case("no_codebook_gs16", rows, cols, gs, None, s, seed=12))

    # 3. group_size 32 (a multiple of 16) and 4 (a divisor): both are legal per
    #    `validate_w4a8_operands`, and a decode that hardcodes 16 gets the scale from
    #    the wrong group for every element past the first.
    #    The scales are large enough that the decoded levels span most of the int8
    #    range: at s_rel ~ 1 the frozen LUT decodes to 0 and ±1 for every group, which
    #    is a case a wrong group index can still pass.
    rows, cols, gs = 3, 256, 32
    s = (16.0 + np.arange(rows * (cols // gs), dtype=np.float32).reshape(rows, -1) * 8.0)
    cases.append(make_case("gs32", rows, cols, gs, lut, s.astype(np.float32), seed=13))

    rows, cols, gs = 2, 256, 4
    s = (8.0 + np.arange(rows * (cols // gs), dtype=np.float32).reshape(rows, -1) * 1.0)
    cases.append(make_case("gs4", rows, cols, gs, lut, s.astype(np.float32), seed=14))

    # 4. Saturation: a huge group scale drives every level past ±127, so the clamp is
    #    what produces the answer rather than the rounding.
    rows, cols, gs = 2, 256, 16
    s = np.full((rows, cols // gs), 256.0, dtype=np.float32)
    cases.append(make_case("saturate", rows, cols, gs, lut, s, seed=15))

    # 5. Ties. A codebook of half-integers at scale 1.0 puts every level exactly on a
    #    .5 boundary, where round-half-to-even and round-half-away-from-zero disagree
    #    on 8 of the 16 (0.5 -> 0 vs 1, 2.5 -> 2 vs 3, ...). Real weights hit this
    #    rarely; nothing else in this fixture can see the difference.
    tie_cb = [float(v) + 0.5 for v in range(-8, 8)]
    rows, cols, gs = 2, 256, 16
    s = np.ones((rows, cols // gs), dtype=np.float32)
    cases.append(make_case("ties_to_even", rows, cols, gs, tie_cb, s, seed=16))

    return cases


def read_tensor(f, base, header, name):
    info = header[name]
    off = info["data_offsets"]
    f.seek(base + off[0])
    return f.read(off[1] - off[0]), info["shape"], info["dtype"]


def real_layer_case(path, layer):
    if not os.path.exists(path):
        print(f"  (skipping the real-layer tier: {path} not present)", file=sys.stderr)
        return None
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        base = 8 + n

        wb, wshape, wdt = read_tensor(f, base, header, layer + ".weight")
        sb, sshape, sdt = read_tensor(f, base, header, layer + ".weight_s_rel")
        cb_b, cbshape, _ = read_tensor(f, base, header, layer + ".weight_codebook")
        sc_b, scshape, _ = read_tensor(f, base, header, layer + ".weight_s_channel")
        qb, _, _ = read_tensor(f, base, header, layer + ".comfy_quant")

    assert wdt == "I8" and sdt == "F8_E4M3", (wdt, sdt)
    conf = json.loads(bytes(qb).decode("utf-8"))
    group_size = int(conf["group_size"])
    rows, k_half = wshape
    cols = k_half * 2

    packed = torch.from_numpy(np.frombuffer(wb, dtype=np.int8).reshape(rows, k_half).copy())
    s_rel = torch.from_numpy(
        np.frombuffer(sb, dtype=np.uint8).reshape(sshape).copy()
    ).view(torch.float8_e4m3fn)
    codebook = torch.from_numpy(np.frombuffer(cb_b, dtype=np.float32).copy())
    s_channel = np.frombuffer(sc_b, dtype=np.float32)

    got = _dequant_int4_grouped_to_int8(packed, s_rel, codebook, group_size).numpy()

    return {
        "layer": layer,
        "quant_conf": conf,
        "rows": rows,
        "cols": cols,
        "group_size": group_size,
        "convrot_groupsize": int(conf["convrot_groupsize"]),
        "codebook": [float(v) for v in codebook.numpy()],
        # Hash the inputs, not the 8.8 GB file: this ties the expectation to exactly
        # the bytes the test reads, so a changed checkpoint is reported as such
        # instead of silently compared against stale weights.
        "packed_fnv1a64": fnv_np(np.frombuffer(wb, dtype=np.uint8)),
        "s_rel_fnv1a64": fnv_np(np.frombuffer(sb, dtype=np.uint8)),
        "s_channel_fnv1a64": fnv_np(s_channel),
        "expect_i8_fnv1a64": fnv_np(got.astype(np.int8)),
        # A few whole values as well, so a mismatching hash can be localized instead
        # of just failing.
        "expect_i8_head": [int(v) for v in got.reshape(-1)[:32]],
        "expect_i8_tail": [int(v) for v in got.reshape(-1)[-32:]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--layer", default=REAL_LAYER)
    args = ap.parse_args()

    torch.manual_seed(0)
    banner = (
        "Generated by tools/gen_w4a8_fixtures.py from comfy_kitchen's own "
        "asym_w4a8_int8 reference. Do not hand-edit."
    )
    out = {
        "_comment": banner,
        "fixed_lut": [float(v) for v in _FIXED_LUT],
        "cases": synthetic_cases(),
    }
    path = os.path.normpath(OUT)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB, "
          f"{len(out['cases'])} synthetic cases)")

    real = real_layer_case(args.checkpoint, args.layer)
    if real is not None:
        real["_comment"] = banner
        real["checkpoint"] = os.path.basename(args.checkpoint)
        rpath = os.path.normpath(OUT_REAL)
        with open(rpath, "w") as f:
            json.dump(real, f, indent=1)
        print(f"wrote {rpath} ({os.path.getsize(rpath) / 1024:.1f} KB, layer {real['layer']})")

    # Teeth check: the ties case must actually distinguish the two rounding modes,
    # else it is decoration. A fixture that cannot fail is not a fixture.
    # Report each case's decoded spread. A case whose levels all collapse to 0/±1 is
    # one a wrong group index or a dropped codebook can still pass, so this is here to
    # keep a weak case from looking like coverage.
    for c in out["cases"]:
        v = np.array(c["expect_i8"], dtype=np.int32)
        print(f"  {c['name']:18s} {len(np.unique(v)):3d} distinct levels, "
              f"range [{v.min():4d}, {v.max():4d}]")
        # `saturate` collapses on purpose (the clamp is what it tests) and
        # `ties_to_even` has only 16 half-integer levels.
        if c["name"] not in ("ties_to_even", "saturate"):
            assert len(np.unique(v)) >= 12, f"{c['name']} decodes to too few levels"

    tie = next(c for c in out["cases"] if c["name"] == "ties_to_even")
    away = [int(np.sign(v) * np.floor(abs(v) + 0.5)) for v in tie["codebook"]]
    even = [int(np.rint(v)) for v in tie["codebook"]]
    differ = sum(1 for a, b in zip(away, even) if a != b)
    assert differ == 8, f"ties case distinguishes only {differ} levels"
    print(f"  ties case: {differ}/16 levels distinguish rint from round-away-from-zero")


if __name__ == "__main__":
    main()
