#!/usr/bin/env python3
"""Pin MiniMax H3's video VAE DECODE by executing ComfyUI's own ViT3D decoder.

Same approach as tools/gen_minimax_h3_forward.py: the real VAE is 5.2 GB, so
build ComfyUI's `MiniMaxH3VideoVAE` at a toy width with seeded weights and record
weights, input and output. What this pins is every convention that differs from
the DiT in the same checkpoint family:

  - `to_qkv` fused PER HEAD (`[h0 q|h0 k|h0 v|h1 q|...]`), not `[all q|all k|all v]`
  - WEIGHTLESS Q/K RMS norms (`elementwise_affine=False`)
  - per-axis token ids in [-1, 1], not the DiT's area-normalized grid
  - partial split-half rope over `int(head_dim * 0.75)`, angle scale 2*pi, base 100
  - four register tokens plus one zero token, dropped after `proj_out`
  - LayerScale residuals (`x += sublayer(...) * scale`)
  - a LayerNorm head where every other norm is RMS
  - the ImageNet output denormalization and clamp

Weights are initialized the way real ones are distributed (norm scales centred on
one, projections fan-in scaled). A uniformly-small init cannot distinguish a
swiglu half swap, which is the mistake the DiT fixture made first; the teeth
check at the end is the guard against repeating it.

Emits src/models/assets/minimax_h3_vae.safetensors.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_minimax_h3_vae.py
"""

import json
import math
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "minimax_h3_vae.safetensors",
)

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.ldm.minimax.vae as h3vae  # noqa: E402
from comfy.ldm.modules.attention import attention_pytorch  # noqa: E402

# See gen_minimax_h3_forward.py: comfy resolves `optimized_attention` at import
# time and lands on xformers, which has no CPU kernel; the module binds the name
# by value, so rebind it here.
h3vae.optimized_attention = attention_pytorch

# head_dim is 64, matching the REAL checkpoint (2048 = 32 x 64). It is the one
# dimension no weight states -- `dim` alone cannot say how it splits -- so
# `Config.detect` takes it as a constant, and a fixture on a different head_dim
# would validate a split the real model never uses. 0.75 of 64 is 48 -> 8
# frequencies per axis -> 24 pairs -> 48 rotated dims, leaving 16 of the head
# unrotated (the partial-rope case). 2 heads so per-head qkv splitting is real.
HEADS = 2
DIM_HEAD = 64
LAYERS = 2
# The REAL patch geometry (16 spatial, 4 temporal), for the same reason as
# head_dim: `proj_out` states only the patch VOLUME, so the split into
# (out_channels, patch_t, patch, patch) is a constant in `Config.detect`, and a
# fixture on a different geometry would validate a split the real model never
# uses. A 2x3x4 latent then decodes to 8 frames of 48x64.
PATCH = 16
PATCH_T = 4
Z_CH = 24
LAT_T, LAT_H, LAT_W = 2, 3, 4


def main():
    torch.manual_seed(20260824)

    dec = h3vae.ViT3DDecoder(
        patch_size=PATCH, patch_size_t=PATCH_T, in_channels=Z_CH, out_channels=3,
        num_layers=LAYERS, heads=HEADS, dim_head=DIM_HEAD,
        rope_theta=100.0, rope_dim_ratio=0.75, num_register_tokens=4,
    )
    dec.eval()

    # From state_dict(), not named_parameters + named_buffers: `pos_embed.inv_freq`
    # is a NON-PERSISTENT buffer (it is derived, not stored), so it is absent from
    # the checkpoint and `load_state_dict(strict=True)` rejects it.
    sd = {}
    for name, p in dec.state_dict().items():
        if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name == "norm_out.weight":
            t = 1.0 + torch.randn(p.shape, dtype=torch.float32) * 0.1
        elif name.startswith("scale"):
            # LayerScale: real ones are small, and a scale of ~1 would make three
            # blocks blow up. Small but not tiny, so the residual still matters.
            t = 0.5 + torch.randn(p.shape, dtype=torch.float32) * 0.1
        elif name.endswith(".bias"):
            t = torch.randn(p.shape, dtype=torch.float32) * 0.1
        elif p.dim() == 2:
            t = torch.randn(p.shape, dtype=torch.float32) / (p.shape[1] ** 0.5)
        else:
            t = torch.randn(p.shape, dtype=torch.float32) * 0.1
        sd[name] = t
    dec.load_state_dict(sd, strict=True)
    for prm in dec.parameters():
        prm.requires_grad_(False)

    # `post_quant_conv` and the latent statistics live on the VAE wrapper, not the
    # decoder, so synthesize them here the way the wrapper holds them.
    post_quant_w = torch.randn(Z_CH, Z_CH, 1, 1, 1, dtype=torch.float32) / (Z_CH ** 0.5)
    post_quant_b = torch.randn(Z_CH, dtype=torch.float32) * 0.1
    latents_mean = torch.randn(Z_CH, dtype=torch.float32) * 0.3
    latents_std = 1.0 + torch.rand(Z_CH, dtype=torch.float32)

    z = torch.randn(1, Z_CH, LAT_T, LAT_H, LAT_W, dtype=torch.float32)

    def run(decoder):
        with torch.inference_mode():
            # the wrapper's decode path for a whole volume: denormalize, 1x1x1
            # post-quant conv, decoder, then the ImageNet finalize
            zz = z * latents_std.view(1, -1, 1, 1, 1) + latents_mean.view(1, -1, 1, 1, 1)
            pq = torch.nn.functional.conv3d(zz, post_quant_w, post_quant_b)
            raw = decoder(pq)
            mean = torch.tensor(h3vae.IMAGENET_MEAN).view(1, 3, 1, 1, 1)
            std = torch.tensor(h3vae.IMAGENET_STD).view(1, 3, 1, 1, 1)
            return (raw * std + mean).clamp_(0.0, 1.0), raw

    out, raw = run(dec)

    assert out.shape == (1, 3, LAT_T * PATCH_T, LAT_H * PATCH, LAT_W * PATCH), out.shape
    assert torch.isfinite(out).all()
    # Clamped output can be degenerate (all 0 or all 1) and still "look" fine;
    # require it to actually vary, and check the pre-clamp field too.
    assert out.std() > 1e-3, float(out.std())
    assert raw.std() > 1e-3, float(raw.std())
    # Clamped pixels carry no information: a fixture that saturates everywhere
    # cannot distinguish anything downstream of the clamp. Require most of the
    # field to be strictly interior.
    interior = float(((out > 1e-4) & (out < 1.0 - 1e-4)).float().mean())
    print("  interior fraction %.3f (raw std %.3f)" % (interior, float(raw.std())))
    assert interior > 0.5, "output is mostly clamped (interior %.3f)" % interior

    # Teeth: swapping the swiglu halves must move the output a lot. The DiT's
    # first fixture could not see this at all.
    orig_ff_forward = h3vae.FeedForward.forward

    def swapped(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(torch.nn.functional.silu(up).mul_(gate))

    h3vae.FeedForward.forward = swapped
    try:
        alt, _ = run(dec)
    finally:
        h3vae.FeedForward.forward = orig_ff_forward
    rel = (alt - out).norm() / out.norm()
    assert rel > 1e-2, "swiglu half order is indistinguishable (rel %.2e)" % rel

    # --- the temporally CHUNKED decode, which is the real decode path -------
    # A whole-volume decode gives t * patch_t frames; the reference's chunking
    # gives fewer (2 latent frames -> 5 pixel frames, not 8) and blends
    # overlapping windows, so the content differs too, not just the count.
    vae = h3vae.MiniMaxH3VideoVAE.__new__(h3vae.MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(vae)
    vae.vae_ratio = PATCH
    vae.vae_ratio_t = PATCH_T
    vae.clip_length = 17
    vae.token_drop = 3
    vae.frame_pre_padding = (-vae.clip_length) % vae.vae_ratio_t
    vae.tokens_chunk_size = math.ceil(vae.clip_length / vae.vae_ratio_t)
    vae.token_overlap = (-vae.token_drop) % vae.tokens_chunk_size
    vae.frame_overlap = max(vae.token_overlap * vae.vae_ratio_t - vae.frame_pre_padding, 0)
    # tiling ON, as the real wrapper defaults: `_adaptive_decode` always goes
    # through `tiled_decode`, so the ViT3D never sees more than tile/patch
    # spatial tokens per axis. A frame at or below `tile_size` is one tile.
    vae.tiling = True
    vae.tile_size = 256
    vae.tile_overlap_min = 64
    vae.decoder = dec
    vae.post_quant_conv = torch.nn.Conv3d(Z_CH, Z_CH, 1)
    with torch.no_grad():
        vae.post_quant_conv.weight.copy_(post_quant_w)
        vae.post_quant_conv.bias.copy_(post_quant_b)
    vae.register_buffer("latents_mean", latents_mean)
    vae.register_buffer("latents_std", latents_std)
    # The pixel statistics are non-persistent buffers on the real wrapper, so
    # __new__ + manual init does not create them.
    vae.register_buffer("pixel_mean", torch.tensor(h3vae.IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
    vae.register_buffer("pixel_std", torch.tensor(h3vae.IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
    for prm in vae.parameters():
        prm.requires_grad_(False)

    # Enough latent frames to make the chunking non-trivial: 7 tokens is two
    # windows, so the blend between them actually runs.
    z_long = torch.randn(1, Z_CH, 7, LAT_H, LAT_W, dtype=torch.float32)
    with torch.inference_mode():
        chunked = vae.decode(z_long)
    want_frames = vae.decode_output_shape(z_long.shape)[2]
    assert chunked.shape[2] == want_frames, (chunked.shape, want_frames)
    assert want_frames != z_long.shape[2] * PATCH_T, "the chunking is a no-op here"
    assert torch.isfinite(chunked).all()
    print("  chunked: %d latent frames -> %d pixel frames (naive would be %d)"
          % (z_long.shape[2], want_frames, z_long.shape[2] * PATCH_T))

    # --- a SPATIALLY TILED decode ------------------------------------------
    # 20x20 latent is 320x320 pixels, which is 2x2 tiles of 256 with a 192 px
    # overlap. The other cases are all below the tile size and so are single
    # tiles, which is exactly why they passed against a NON-tiled decode.
    z_tiled = torch.randn(1, Z_CH, 2, 20, 20, dtype=torch.float32)
    with torch.inference_mode():
        tiled = vae.decode(z_tiled)
    ys = vae.split_tiles(320)
    assert len(ys[0]) > 1, "the tiling case does not actually tile"
    print("  tiled: 320 px -> %d tiles at %s, overlaps %s" % (len(ys[0]), ys[0], ys[2]))
    assert torch.isfinite(tiled).all()

    tensors = {"decoder." + k: v for k, v in sd.items()}
    tensors["in.z_long"] = z_long
    tensors["out.chunked"] = chunked
    tensors["in.z_tiled"] = z_tiled
    tensors["out.tiled"] = tiled
    tensors["post_quant_conv.weight"] = post_quant_w
    tensors["post_quant_conv.bias"] = post_quant_b
    tensors["latents_mean"] = latents_mean
    tensors["latents_std"] = latents_std
    tensors["in.z"] = z
    tensors["out.rgb"] = out

    meta = {
        "config": json.dumps(dict(heads=HEADS, dim_head=DIM_HEAD, layers=LAYERS,
                                  patch=PATCH, patch_t=PATCH_T, z_channels=Z_CH)),
        "shape": json.dumps(dict(t=LAT_T, h=LAT_H, w=LAT_W)),
        "note": "generated by tools/gen_minimax_h3_vae.py from "
                "comfy.ldm.minimax.vae.ViT3DDecoder; do not hand-edit",
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    safetensors.torch.save_file(tensors, OUT, metadata=meta)
    print("wrote %s: %d tensors, %d bytes" % (OUT, len(tensors), os.path.getsize(OUT)))
    print("  out std %.5f  min %.4f max %.4f  swiglu sensitivity %.4f"
          % (float(out.std()), float(out.min()), float(out.max()), rel))


if __name__ == "__main__":
    sys.exit(main())
