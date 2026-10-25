#!/usr/bin/env python3
"""Reference fixtures for Nunchaku's SVDQuant W4A4 weight format, int4 and fp4.

The packed tensors come from Nunchaku's own `NunchakuWeightPacker` run on random plain
values, never from ours. An `SVDQW4A4Linear` (precision int4) ships:

    qweight             I8    [rows, cols/2]   int4 codes, MMA tile order
    wscales             BF16  [cols/64, rows]  per-64-input scale, packed (`pack_scale`)
    smooth_factor       BF16  [cols]           packed like a scale (group size -1)
    proj_down           BF16  [cols, rank]     low-rank down, packed (`pack_lowrank_weight`)
    proj_up             BF16  [rows, rank]     low-rank up, packed

and computes, per Nunchaku's `quantize_w4a4_act_fuse_lora` and `gemm_w4a4` kernels,

    y = Q4(x / smooth) @ (codes * scale)^T  +  (x @ down^T) @ up^T

with the low-rank branch on the RAW x. The fixture's `weff` and `y` drop only the
activation quantization: `weff = codes * scale / smooth`.

The fp4 layer (keys under `fp4.`) stores E2M1 bit patterns in the same word order, an
fp8-e4m3 scale per 16 inputs packed by `pack_micro_scale`, and `wcscales` (packed like a
vector) times a scalar `wtscale` per output row, which multiplies the 4-bit product only:
`weff = e2m1(codes) * scale * wcscales * wtscale / smooth`.

Every packed layout is a permutation that keeps every magnitude, so the generator
asserts each one actually moves values, and that the corpus has a -8 code, a
non-uniform smoothing vector and more than one 128-row tile.

Nunchaku is fetched at generation time and is not vendored; the fixture records its
commit. Its package `__init__` loads the compiled CUDA extension, so the three pure
Python modules the packer needs are loaded directly.

Usage:
    git clone https://github.com/nunchaku-tech/nunchaku /tmp/nunchaku
    /dump/venv-eg2/bin/python -I tools/gen_svdq_fixtures.py /tmp/nunchaku
"""

import importlib.util
import os
import subprocess
import sys
import types

import torch
from safetensors.torch import save_file

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "ops", "assets", "svdq_fixtures.safetensors")

ROWS, COLS, RANK, M = 256, 256, 32, 3


def load_packer(root):
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    for pkg in ("nunchaku", "nunchaku.lora", "nunchaku.lora.flux"):
        m = types.ModuleType(pkg)
        m.__path__ = []
        sys.modules[pkg] = m
    hub = types.ModuleType("huggingface_hub")  # imported by nunchaku/utils.py, unused by the packer
    hub.hf_hub_download = None
    sys.modules["huggingface_hub"] = hub
    load("nunchaku.utils", f"{root}/nunchaku/utils.py")
    load("nunchaku.lora.flux.utils", f"{root}/nunchaku/lora/flux/utils.py")
    return load("nunchaku.lora.flux.packer", f"{root}/nunchaku/lora/flux/packer.py").NunchakuWeightPacker(bits=4)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def fp4_layer(p, g):
    codes = torch.randint(0, 16, (ROWS, COLS), generator=g, dtype=torch.int32)
    # Scales already on the fp8 grid, so the packer's own cast to fp8 is exact.
    scales = (torch.rand(ROWS, COLS // 16, generator=g) * 3 + 0.05).to(torch.float8_e4m3fn).to(torch.bfloat16)
    smooth = torch.exp(torch.randn(COLS, generator=g) * 0.7).to(torch.bfloat16)
    wc = (torch.rand(ROWS, generator=g) * 0.02 + 0.001).to(torch.bfloat16)
    wtscale = torch.tensor([0.75])
    down = (torch.randn(RANK, COLS, generator=g) * 0.05).to(torch.bfloat16)
    up = (torch.randn(ROWS, RANK, generator=g) * 0.05).to(torch.bfloat16)
    x = torch.randn(M, COLS, generator=g)

    qweight = p.pack_weight(codes.clone())
    wscales = p.pack_micro_scale(scales.clone(), group_size=16)
    wc_p = p.pack_scale(wc.clone(), group_size=-1).view(-1)
    smooth_p = p.pack_scale(smooth.clone(), group_size=-1).view(-1)

    for c in (7, 8, 15):
        assert (codes == c).any(), f"corpus lacks E2M1 pattern {c}"
    assert not torch.equal(wscales.view(torch.uint8).reshape(-1), scales.to(torch.float8_e4m3fn).t().contiguous().view(torch.uint8).reshape(-1)), "micro-scale packing is a transpose"
    assert not torch.equal(wc_p, wc), "wcscales packing is the identity"
    assert wc.float().std() > 0, "wcscales constant: its order would be untested"

    row_mul = wc.float() * wtscale
    weff = E2M1[codes.long()] * scales.float().repeat_interleave(16, dim=1) * row_mul[:, None] / smooth.float()[None, :]
    xd = x.double()
    y = (xd @ weff.double().t() + (xd @ down.double().t()) @ up.double().t()).float()
    return {
        "fp4.qweight": qweight.contiguous(),
        "fp4.wscales": wscales.contiguous(),
        "fp4.wcscales": wc_p.contiguous(),
        "fp4.wtscale": wtscale.contiguous(),
        "fp4.smooth_factor": smooth_p.contiguous(),
        "fp4.proj_down": p.pack_lowrank_weight(down.clone(), down=True).contiguous(),
        "fp4.proj_up": p.pack_lowrank_weight(up.clone(), down=False).contiguous(),
        "fp4.codes": codes.to(torch.int8).contiguous(),
        "fp4.scales": scales.float().contiguous(),
        "fp4.smooth": smooth.float().contiguous(),
        "fp4.row_mul": row_mul.contiguous(),
        "fp4.down": down.contiguous(),
        "fp4.up": up.contiguous(),
        "fp4.weff": weff.contiguous(),
        "fp4.x": x.contiguous(),
        "fp4.y": y.contiguous(),
    }


def main():
    root = sys.argv[1]
    commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    p = load_packer(root)
    g = torch.Generator().manual_seed(1234)

    codes = torch.randint(-8, 8, (ROWS, COLS), generator=g, dtype=torch.int32)
    scales = (torch.rand(ROWS, COLS // 64, generator=g) * 0.02 + 0.001).to(torch.bfloat16)
    smooth = torch.exp(torch.randn(COLS, generator=g) * 0.7).to(torch.bfloat16)
    down = (torch.randn(RANK, COLS, generator=g) * 0.05).to(torch.bfloat16)
    up = (torch.randn(ROWS, RANK, generator=g) * 0.05).to(torch.bfloat16)
    x = torch.randn(M, COLS, generator=g)

    qweight = p.pack_weight(codes.clone())
    wscales = p.pack_scale(scales.clone(), group_size=64)
    smooth_p = p.pack_scale(smooth.clone(), group_size=-1).view(-1)
    down_p = p.pack_lowrank_weight(down.clone(), down=True)
    up_p = p.pack_lowrank_weight(up.clone(), down=False)

    # The packer's own inverse where it has one, so a wrong lowrank orientation here
    # cannot pass.
    assert torch.equal(p.unpack_lowrank_weight(down_p, down=True), down)
    assert torch.equal(p.unpack_lowrank_weight(up_p, down=False), up)

    assert (codes == -8).any() and (codes == 7).any(), "corpus lacks a -8 or a 7 code"
    assert smooth.float().std() > 0.3, "smoothing too uniform to tell a wrong order"
    assert ROWS > 128, "one tile cannot show the tile order"
    nib = lambda b: torch.stack([b & 0xF, (b >> 4) & 0xF], -1).reshape(ROWS, COLS)
    assert not torch.equal(nib(qweight.view(torch.uint8).int()), codes & 0xF), "qweight packing is the identity"
    assert not torch.equal(wscales.view(-1), scales.t().contiguous().view(-1)), "scale packing is the identity"
    assert not torch.equal(smooth_p, smooth), "vector packing is the identity"
    assert not torch.equal(down_p, down.t().contiguous()), "down packing is a transpose"

    s32 = scales.float().repeat_interleave(64, dim=1)
    weff = codes.float() * s32 / smooth.float()[None, :]
    xd = x.double()
    y = (xd @ weff.double().t() + (xd @ down.double().t()) @ up.double().t()).float()

    fp4 = fp4_layer(p, g)
    tensors = {
        **fp4,
        "qweight": qweight.contiguous(),
        "wscales": wscales.contiguous(),
        "smooth_factor": smooth_p.contiguous(),
        "proj_down": down_p.contiguous(),
        "proj_up": up_p.contiguous(),
        "codes": codes.to(torch.int8).contiguous(),
        "scales": scales.float().contiguous(),
        "smooth": smooth.float().contiguous(),
        "down": down.contiguous(),
        "up": up.contiguous(),
        "weff": weff.contiguous(),
        "x": x.contiguous(),
        "y": y.contiguous(),
    }
    save_file(tensors, OUT, metadata={"nunchaku_commit": commit, "generator": "tools/gen_svdq_fixtures.py"})
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes), nunchaku {commit}")


if __name__ == "__main__":
    main()
