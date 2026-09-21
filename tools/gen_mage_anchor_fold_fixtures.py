#!/usr/bin/env python3
"""Reference fixtures for the Flux2-anchored latent fold (`sd_vae.PackedLatent`).

Reference is **ComfyUI**'s `AutoencodingEngineLegacy.decode`, whose first two
statements undo a BatchNorm over the folded channels and then unfold each 2x2
pixel block back out of the channel axis. That pair is what lets Mage-Flow decode
through Flux 2's VAE at all, and it is the only new arithmetic on that path: the
`AutoencoderKL` underneath is the one `sd_vae` already runs for Z-Image.

This does NOT reimplement the fold. It builds the real engine (at toy net widths,
since the adapter does not depend on them), replaces `post_quant_conv` and
`decoder` with identities, and calls the real `decode`, so the einops pattern and
the eps come from the installed reference rather than from a transcription here.

The cases separate what a wrong fold still gets right:

  - `even` (4x6 latent): the plain case.
  - `odd` (3x5 latent): both extents odd, so a reading that transposes the two
    sub-pixel axes lands on a different pixel instead of a symmetric one.

`bn_mean` / `bn_var` are drawn with a spread wide enough that dropping the
BatchNorm, or applying it with torch's default 1e-5 eps instead of ComfyUI's
1e-4, changes the output by more than the f32 comparison tolerance.

Run it memory-bounded:

    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_mage_anchor_fold_fixtures.py
"""

import os
import sys

COMFY = "/home/qt/genai/comfyui"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_OUT = os.path.join(REPO, "src", "models", "assets", "mage_anchor_fold_ref.safetensors")

SEED = 20260919
Z_CHANNELS = 32
# (name, latent h, latent w) at the FOLDED resolution the DiT works in.
CASES = [("even", 4, 6), ("odd", 3, 5)]

sys.argv = [sys.argv[0], "--cpu", "--fp32-vae", "--disable-smart-memory"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
import safetensors.torch  # noqa: E402
from comfy.cli_args import args as comfy_args  # noqa: E402
from comfy.ldm.models.autoencoder import AutoencoderKL  # noqa: E402

if not comfy_args.cpu:
    raise SystemExit("ComfyUI did not take our flags — check comfy.options.enable_args_parsing()")


def build_engine():
    """The real engine with `batch_norm_latent`, at toy net widths.

    `sd.py` sets `batch_norm_latent` from the presence of `bn.running_mean` and
    nothing else, so the flag is the whole configuration of the fold.
    """
    ddconfig = {
        "double_z": True,
        "z_channels": Z_CHANNELS,
        "resolution": 64,
        "in_channels": 3,
        "out_ch": 3,
        # GroupNorm's 32 groups sets the floor; the fold does not read these.
        "ch": 32,
        "ch_mult": [1],
        "num_res_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.0,
        "batch_norm_latent": True,
    }
    eng = AutoencoderKL(ddconfig=ddconfig, embed_dim=Z_CHANNELS)
    eng.eval()
    if eng.bn is None:
        raise SystemExit("batch_norm_latent did not take; the reference moved")
    if list(eng.ps) != [2, 2]:
        raise SystemExit(f"unexpected fold {eng.ps}; this fixture pins 2x2")
    if eng.bn.num_features != 4 * Z_CHANNELS:
        raise SystemExit("bn width is not fold^2 * z_channels")
    # Everything after the fold is the ordinary AutoencoderKL that `sd_vae`
    # already covers, so cut it out and let `decode` return the unfolded latent.
    eng.post_quant_conv = torch.nn.Identity()
    eng.decoder = _PassThrough()
    return eng


class _PassThrough(torch.nn.Module):
    """`decode` calls the decoder with kwargs; `nn.Identity` refuses them."""

    def forward(self, x, **kwargs):
        return x


def main():
    torch.manual_seed(SEED)
    eng = build_engine()
    packed_ch = 4 * Z_CHANNELS

    # Wide enough that a missing or mis-eps'd BatchNorm is not lost in tolerance.
    mean = torch.randn(packed_ch, dtype=torch.float32) * 0.5
    var = torch.rand(packed_ch, dtype=torch.float32) * 3.0 + 0.25
    eng.bn.running_mean.copy_(mean)
    eng.bn.running_var.copy_(var)

    out = {"bn_mean": mean, "bn_var": var}
    for name, lh, lw in CASES:
        z = torch.randn(1, packed_ch, lh, lw, dtype=torch.float32)
        with torch.no_grad():
            y = eng.decode(z)
        assert y.shape == (1, Z_CHANNELS, lh * 2, lw * 2), y.shape
        out[f"{name}.z"] = z[0].contiguous()
        out[f"{name}.out"] = y[0].contiguous()

        # Teeth. Each of these is a fold a careless port actually writes, and
        # each must move the answer: if one does not, the case cannot see it.
        s = torch.sqrt(var.view(1, -1, 1, 1) + eng.bn_eps)
        m = mean.view(1, -1, 1, 1)
        zb = z * s + m
        # (a) sub-pixel axes swapped: `(c pj pi)` instead of `(c pi pj)`.
        swapped = zb.view(1, Z_CHANNELS, 2, 2, lh, lw).permute(0, 1, 5, 3, 4, 2)
        swapped = swapped.reshape(1, Z_CHANNELS, lw * 2, lh * 2)
        if swapped.shape == y.shape and torch.allclose(swapped, y, atol=1e-5):
            raise SystemExit(f"{name}: axis-swapped fold is indistinguishable")
        # (b) channel fastest: `(pi pj c)` instead of `(c pi pj)`.
        chan_fast = zb.view(1, 2, 2, Z_CHANNELS, lh, lw).permute(0, 3, 4, 1, 5, 2)
        chan_fast = chan_fast.reshape(1, Z_CHANNELS, lh * 2, lw * 2)
        if torch.allclose(chan_fast, y, atol=1e-5):
            raise SystemExit(f"{name}: channel-fastest fold is indistinguishable")
        # (c) BatchNorm dropped entirely.
        no_bn = z.view(1, Z_CHANNELS, 2, 2, lh, lw).permute(0, 1, 4, 2, 5, 3)
        no_bn = no_bn.reshape(1, Z_CHANNELS, lh * 2, lw * 2)
        if torch.allclose(no_bn, y, atol=1e-5):
            raise SystemExit(f"{name}: the BatchNorm does not move this case")
        # (d) torch's default eps instead of ComfyUI's 1e-4.
        wrong_eps = z * torch.sqrt(var.view(1, -1, 1, 1) + 1e-5) + m
        wrong_eps = wrong_eps.view(1, Z_CHANNELS, 2, 2, lh, lw).permute(0, 1, 4, 2, 5, 3)
        wrong_eps = wrong_eps.reshape(1, Z_CHANNELS, lh * 2, lw * 2)
        if torch.allclose(wrong_eps, y, atol=1e-6):
            raise SystemExit(f"{name}: eps 1e-5 vs 1e-4 does not move this case")

    os.makedirs(os.path.dirname(REF_OUT), exist_ok=True)
    safetensors.torch.save_file(
        out,
        REF_OUT,
        metadata={
            "bn_eps": repr(eng.bn_eps),
            "fold": "2",
            "z_channels": str(Z_CHANNELS),
            "cases": ",".join(f"{n}:{h}x{w}" for n, h, w in CASES),
            "seed": str(SEED),
        },
    )
    print(f"wrote {REF_OUT}")
    for k, v in out.items():
        print(f"  {k:12s} {tuple(v.shape)}")


if __name__ == "__main__":
    main()
