#!/usr/bin/env python3
"""Pin MiniMax H3's video VAE ENCODE by executing ComfyUI's own encoder.

The decode side is a ViT3D; the encode side is a completely different network, a
3-D causal CNN, so nothing in the existing video-VAE fixture covers it. What this
pins is the convention set that is silent when wrong:

  - CausalConv3d: REFLECT spatial padding, CAUSAL temporal padding, and the front
    pad is `padding[0] * 2` frames, not `padding[0]`.
  - the SINGLE-FRAME path truncates the temporal taps instead of convolving zero
    frames (`autopad="causal_zero"`), and then keeps only the LAST temporal slice.
    That is the whole fl2va / ref-image case, so it is the load-bearing one.
  - TemporalIsolatedGroupNorm computes statistics PER FRAME (time folded into the
    batch). A plain 3-D GroupNorm over (C,T,H,W) is finite and wrong.
  - Downsample3D: asymmetric reflect pad (0,1,0,1) THEN a stride-2 conv whose own
    padding is (1,0,0), i.e. temporal only.
  - ResnetBlock3D is pre-norm (norm -> silu -> conv, twice) with a 1x1x1
    `nin_shortcut` only when the channel count changes.
  - `conv_out` emits 2 * z_channels and `quant_conv` is a 1x1x1 Conv3d on top;
    encode keeps the FIRST chunk (the mean) and drops the log-variance.
  - pixel normalization is `(x + 1) / 2` then ImageNet mean/std, and the latent is
    `(mean - latents_mean) / latents_std` per channel.
  - SPATIAL TILING IS ALWAYS ON (`tiling=True` by default) at 256 px with a 64 px
    minimum overlap. One case here is deliberately WIDER than a tile, because a
    frame at or below the tile size is exactly one tile and would pass either way.
    That hole is what made the decode port ship with a visible patch grid.

⚠️ The toy uses **8 GroupNorm groups, not the real 32**, and that is the one thing
this fixture does NOT pin. It cannot be both small and 32-grouped: 32 groups force
every level's channel count to a multiple of 32, which at six levels is
32/64/64/128/128/256 and tens of MB of weights. The group count is not derivable
from a checkpoint either (a GroupNorm weight is `[C]` whatever the grouping), so it
is an architecture constant on our side, and `minimax_h3_vae.zig` carries it as a
Config field with the real default. A separate test asserts that default and that
the field is actually honoured rather than hardcoded downstream.

Emits src/models/assets/minimax_h3_vae_encode.safetensors.

Usage (BOUND IT -- this builds a model):
    systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_minimax_h3_vae_encode.py
"""

import json
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "minimax_h3_vae_encode.safetensors",
)

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.ldm.minimax.vae as mvae  # noqa: E402

# The level COUNT and the down/up schedule are the real ones, because those are
# what is structurally load-bearing. Only `ch` and the group count shrink.
CH = 8
CH_MULT = (1, 2, 2, 4, 4, 8)
SPACE_DOWN = (2, 2, 2, 2, 1, 1)
TIME_DOWN = (1, 2, 2, 1, 1, 1)
NUM_RES = 2
Z_CH = 24
EMBED = 24
TOY_GROUPS = 8
REAL_GROUPS = 32

# (name, T, H, W). The 288-wide case is what makes the tiling real: 288 > 256, so
# it splits into two tiles and blends, where 256 and below are a single tile.
CASES = [
    ("one_frame_small", 1, 64, 64),
    ("one_frame_tiled", 1, 128, 288),
    ("clip_5", 5, 64, 64),
]


def main():
    torch.manual_seed(20260827)
    torch.set_num_threads(4)

    levels = [CH * m for m in CH_MULT]
    assert all(c % TOY_GROUPS == 0 for c in levels), levels
    print("toy: ch %d, groups %d, levels %s" % (CH, TOY_GROUPS, levels))

    # Two things have to be patched around the constructor, both because the
    # reference hardcodes them rather than taking them as parameters.
    #
    # 1. `group_norm_3d` fixes 32 groups, and 32 groups force every level's
    #    channel count to a multiple of 32.
    # 2. `ViT3DDecoder` is built at its REAL size (36 blocks x 2048, ~67M
    #    parameters each) whatever `ch` is, so constructing the whole VAE to get
    #    at its encoder costs 2.4 BILLION parameters and 9.2 GB. `encode` never
    #    touches the decoder, and the decode side has its own fixture
    #    (minimax_h3_vae.safetensors), so a stub is exact here rather than a
    #    shortcut.
    class _NoDecoder(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()

    real_group_norm = mvae.group_norm_3d
    real_decoder = mvae.ViT3DDecoder
    mvae.group_norm_3d = lambda num_channels: mvae.TemporalIsolatedGroupNorm(
        num_groups=TOY_GROUPS, num_channels=num_channels, eps=1e-6, affine=True)
    mvae.ViT3DDecoder = _NoDecoder
    try:
        vae = mvae.MiniMaxH3VideoVAE(
            ch=CH, embed_dim=EMBED, z_channels=Z_CH, ch_mult=CH_MULT,
            num_res_blocks=NUM_RES, space_down=SPACE_DOWN, time_down=TIME_DOWN,
        )
    finally:
        mvae.group_norm_3d = real_group_norm
        mvae.ViT3DDecoder = real_decoder
    vae.eval()
    n_params = sum(p.numel() for p in vae.parameters())
    print("  %d parameters (%.1f MB f32)" % (n_params, n_params * 4 / 1048576))
    # A guard, not a nicety: the first version of this script built the real
    # decoder by accident and came to 2.4e9 parameters / 9.2 GB.
    assert n_params < 5_000_000, "the 'toy' model is not toy: %d parameters" % n_params

    # Seeded weights with a sane scale: six levels deep, and a unit-variance init
    # would saturate the GroupNorms long before conv_out.
    sd = {}
    for name, p in vae.state_dict().items():
        if name in ("latents_mean", "latents_std"):
            sd[name] = p.clone()
        elif p.dim() == 5:  # conv weight [out, in, kt, kh, kw]
            fan_in = p.shape[1] * p.shape[2] * p.shape[3] * p.shape[4]
            sd[name] = torch.randn(p.shape, dtype=torch.float32) / (fan_in ** 0.5)
        elif name.endswith(".weight"):  # groupnorm gamma
            sd[name] = 1.0 + torch.randn(p.shape, dtype=torch.float32) * 0.1
        else:
            sd[name] = torch.randn(p.shape, dtype=torch.float32) * 0.05
    vae.load_state_dict(sd, strict=True)
    for prm in vae.parameters():
        prm.requires_grad_(False)

    # Only the ENCODE half is needed; dropping the decoder keeps the fixture from
    # carrying a second network nothing here tests.
    tensors = {}
    for k, v in sd.items():
        if k.startswith("decoder.") or k.startswith("post_quant_conv."):
            continue
        tensors["vae." + k] = v

    meta_cases = {}
    for name, t, h, w in CASES:
        x = torch.randn(1, 3, t, h, w, dtype=torch.float32).clamp(-1, 1)
        with torch.inference_mode():
            z = vae.encode(x)
        assert z.shape[0] == 1 and z.shape[1] == Z_CH, z.shape
        assert z.shape[3] == h // 16 and z.shape[4] == w // 16, (z.shape, h, w)
        assert torch.isfinite(z).all()
        assert z.std() > 1e-3, (name, float(z.std()))
        tensors["in.%s" % name] = x
        tensors["out.%s" % name] = z
        meta_cases[name] = dict(t=t, h=h, w=w, latent_t=int(z.shape[2]),
                                latent_h=int(z.shape[3]), latent_w=int(z.shape[4]))
        print("  %-16s [1,3,%d,%d,%d] -> %s  std %.4f"
              % (name, t, h, w, list(z.shape), float(z.std())))

    # A single frame must give latent_t == 1, and a 5-frame clip must NOT: the
    # temporal ratio is 4 and `token_drop` trims the tail, so these two paths
    # produce different lengths and a port that ran one for the other would be
    # caught by shape alone.
    assert meta_cases["one_frame_small"]["latent_t"] == 1, meta_cases
    assert meta_cases["clip_5"]["latent_t"] != 1, meta_cases

    # --- teeth -------------------------------------------------------------
    small = tensors["in.one_frame_small"]
    ref = tensors["out.one_frame_small"]

    def rel(a, b):
        return float((a - b).norm() / b.norm())

    import torch.nn.functional as F

    # 1. ZERO spatial padding instead of reflect.
    orig_fwd = mvae.CausalConv3d.forward

    def zero_pad_fwd(self, x):
        if sum(self.causal_padding) == 0:
            return super(mvae.CausalConv3d, self).forward(x)
        x = F.pad(x, (self.causal_padding[2], self.causal_padding[2],
                      self.causal_padding[1], self.causal_padding[1], 0, 0), mode="constant")
        if x.shape[2] == 1:
            return super(mvae.CausalConv3d, self).forward(x, autopad="causal_zero")
        x = F.pad(x, (0, 0, 0, 0, self.causal_padding[0] * 2, 0), mode="constant")
        return super(mvae.CausalConv3d, self).forward(x)

    mvae.CausalConv3d.forward = zero_pad_fwd
    try:
        with torch.inference_mode():
            alt = vae.encode(small)
    finally:
        mvae.CausalConv3d.forward = orig_fwd
    d_pad = rel(alt, ref)
    assert d_pad > 1e-2, "reflect vs zero spatial padding is indistinguishable (%.2e)" % d_pad

    # 2. GroupNorm over the whole volume instead of per frame. Only a multi-frame
    #    case can see this at all, which is why `clip_5` exists.
    orig_gn = mvae.TemporalIsolatedGroupNorm.forward
    mvae.TemporalIsolatedGroupNorm.forward = lambda self, x: torch.nn.GroupNorm.forward(self, x)
    try:
        with torch.inference_mode():
            alt = vae.encode(tensors["in.clip_5"])
            alt1 = vae.encode(small)
    finally:
        mvae.TemporalIsolatedGroupNorm.forward = orig_gn
    d_gn = rel(alt, tensors["out.clip_5"])
    assert d_gn > 1e-2, "per-frame vs whole-volume GroupNorm is indistinguishable (%.2e)" % d_gn
    d_gn1 = rel(alt1, ref)
    assert d_gn1 < 1e-6, "a single frame should be blind to the GroupNorm axis (%.2e)" % d_gn1

    # 3. Taking the log-variance half instead of the mean.
    with torch.inference_mode():
        moments = vae._adaptive_encode(vae._normalize_pixels(small))[:, :, -1:, :, :]
        logvar = torch.chunk(moments.float(), 2, dim=1)[1]
        lm = vae.latents_mean.view(1, -1, 1, 1, 1)
        ls = vae.latents_std.view(1, -1, 1, 1, 1)
        wrong_half = (logvar - lm) / ls
    d_half = rel(wrong_half, ref)
    assert d_half > 1e-2, "mean vs logvar half is indistinguishable (%.2e)" % d_half

    # 4. Tiling actually splits the wide case, and actually does NOT split the
    #    small one. Both halves matter: the first says the fixture has teeth for
    #    tiling, the second says 64px is a legitimate single-tile control.
    vae.tiling = False
    try:
        with torch.inference_mode():
            untiled_wide = vae.encode(tensors["in.one_frame_tiled"])
            untiled_small = vae.encode(small)
    finally:
        vae.tiling = True
    d_tile = rel(untiled_wide, tensors["out.one_frame_tiled"])
    assert d_tile > 1e-3, "the 288-wide case does not actually tile (%.2e)" % d_tile
    d_small = rel(untiled_small, ref)
    assert d_small < 1e-6, "the 64px case should be ONE tile, but tiling changed it (%.2e)" % d_small

    meta = {
        "config": json.dumps(dict(ch=CH, ch_mult=list(CH_MULT), space_down=list(SPACE_DOWN),
                                  time_down=list(TIME_DOWN), num_res_blocks=NUM_RES,
                                  z_channels=Z_CH, embed_dim=EMBED,
                                  norm_groups=TOY_GROUPS, real_norm_groups=REAL_GROUPS,
                                  tile_size=256, tile_overlap_min=64,
                                  clip_length=17, token_drop=3)),
        "cases": json.dumps(meta_cases),
        "note": "generated by tools/gen_minimax_h3_vae_encode.py from "
                "comfy.ldm.minimax.vae.MiniMaxH3VideoVAE.encode; do not hand-edit. "
                "norm_groups is 8 here and 32 in the real checkpoint; see the header.",
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    safetensors.torch.save_file(tensors, OUT, metadata=meta)
    print("wrote %s: %d tensors, %.2f MB" % (OUT, len(tensors), os.path.getsize(OUT) / 1048576))
    print("  sensitivity: reflect-pad %.4f  per-frame-GN %.4f  mean-half %.4f  tiling %.4f"
          % (d_pad, d_gn, d_half, d_tile))


if __name__ == "__main__":
    sys.exit(main())
