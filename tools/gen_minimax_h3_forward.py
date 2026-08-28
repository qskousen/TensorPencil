#!/usr/bin/env python3
"""Pin MiniMax H3's DiT forward by EXECUTING ComfyUI's own model at a tiny width.

The real checkpoint is 21 GB of int8 and cannot be a unit fixture. What CAN be
pinned exactly is the ARCHITECTURE: build ComfyUI's `MiniMaxH3Model` at a toy
width with seeded random f32 weights, run it, and record the weights, inputs and
outputs. Every convention that makes H3 what it is -- the three-tag adaLN row
layout, the partial split-half rope over an area-normalized position grid, the
non-uniform video time axis, channel-major audio, the curve-interpolated time
embedding, the swiglu half order, the negated output -- is exercised at this
width exactly as at the real one.

What this does NOT pin is the int8 convrot decode or the real weights; the gated
`Config.detect` test covers the checkpoint's shapes, and a real-render parity
test against ComfyUI is a separate (much more expensive) thing.

Emits src/models/assets/minimax_h3_forward.safetensors, read by the forward
parity test in src/models/minimax_h3.zig.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_minimax_h3_forward.py
"""

import json
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "minimax_h3_forward.safetensors",
)

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.ops  # noqa: E402
import comfy.ldm.minimax.model as h3  # noqa: E402
from comfy.ldm.modules.attention import attention_pytorch  # noqa: E402
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout  # noqa: E402

# ComfyUI resolves its attention backend at IMPORT time and checks xformers
# before pytorch, so it lands on xformers, which has no CPU kernel at all (it
# refuses on `device=cpu` whatever the head width). The CLI flags do not help
# here because the model module binds the name by value
# (`from ... import optimized_attention`), so rebind that name rather than the
# module it came from. This fixture is about the architecture, not the kernel.
h3.optimized_attention = attention_pytorch

# A toy width that still exercises every structural choice:
#  - 2 heads so the per-head Q/K norm is not a whole-row norm
#  - head_dim 32 with rope_inv_freq_len 4 => 24 rotated dims of 32, so the
#    UNROTATED tail is non-empty (the partial-rope case). 32 is also the floor
#    xformers will dispatch at on the CPU.
#  - inner (heads*head_dim = 64) != hidden (32), as in the real model
#    (7168 vs 5376): out_proj is rectangular, and a square one would hide it
#  - 3 trunk blocks and 2 refiner blocks: enough for a residual chain to diverge
CFG = dict(
    hidden_size=32,
    num_layers=3,
    token_refiner_num_layers=2,
    num_attention_heads=2,
    attention_head_dim=32,
    ffn_hidden_size=48,
    latents_dim=24,
    audio_latents_dim=32,
    patch_size=(1, 2, 2),
    text_dim=20,
    time_embed_dim=8,
    adaln_curve_grid=33,
    rope_inv_freq_len=4,
)

TEXT_LEN = 4
LATENT_T = 2
LATENT_H = 4
LATENT_W = 6
AUDIO_T = 3
SIGMA = 0.7


def main():
    torch.manual_seed(20260824)
    ops = comfy.ops.disable_weight_init

    model = MiniMaxH3Model(operations=ops, dtype=torch.float32, **CFG)
    model.eval()

    # Seeded weights, initialized the way REAL ones are distributed rather than
    # uniformly small. This is what gives the fixture teeth.
    #
    # A first version used `randn * 0.05` throughout and could not distinguish
    # the swiglu half order at all (swapping `silu(gate)*value` for
    # `silu(value)*gate` moved the output by 3e-7, against a 1e-5 threshold).
    # Two reasons, both fixed here: RMSNorm scales drawn around ZERO annihilate
    # the signal, and activations that never leave silu's near-linear region make
    # `silu(a)*b` and `silu(b)*a` agree to second order.
    sd = {}
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        if name == "rope.inv_freq":
            # the real checkpoint's inv_freq is a decaying positive ladder; a
            # random one would still work but this keeps the angles realistic
            t = 1.0 / (10000.0 ** (torch.arange(p.numel(), dtype=torch.float32) / p.numel()))
        elif name == "adaln_t_table":
            # a smooth curve, so the lerp in `timeEmbed` is actually interpolating
            # between distinguishable rows rather than between noise
            grid, dim = p.shape
            ramp = torch.linspace(0.0, 1.0, grid).unsqueeze(1)
            phase = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
            t = torch.sin(ramp * 3.0 + phase) * 2.0
        elif name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
            # every RMSNorm scale, per-head Q/K norms included: centred on ONE,
            # like a trained norm, so the residual stream survives the block
            t = 1.0 + torch.randn(p.shape, dtype=torch.float32) * 0.1
        elif name.endswith("adaln_proj.linear.weight"):
            # modulation: keep (1 + scale) near 1 so three blocks stay bounded
            t = torch.randn(p.shape, dtype=torch.float32) * 0.1
        elif name.endswith(".bias"):
            t = torch.randn(p.shape, dtype=torch.float32) * 0.1
        elif p.dim() == 2:
            # unit-variance outputs: fan-in scaling, so activations land in the
            # range where silu is genuinely nonlinear
            t = torch.randn(p.shape, dtype=torch.float32) / (p.shape[1] ** 0.5)
        else:
            t = torch.randn(p.shape, dtype=torch.float32) * 0.1
        sd[name] = t
    model.load_state_dict(sd, strict=True)
    # The fused in-place RMSNorm+rope kernels refuse any tensor with
    # `requires_grad` set (`check_rope_inplace`), and the WEIGHTS reach them as
    # the norm scales. Parameters default to requires_grad, so clear it: neither
    # no_grad nor inference_mode changes the flag on a parameter.
    for prm in model.parameters():
        prm.requires_grad_(False)

    # Inputs. Batch 1 is the only supported batch.
    video = torch.randn(1, CFG["latents_dim"], LATENT_T, LATENT_H, LATENT_W, dtype=torch.float32)
    audio = torch.randn(1, CFG["audio_latents_dim"], 2, AUDIO_T, dtype=torch.float32)
    text = torch.randn(1, TEXT_LEN, CFG["text_dim"], dtype=torch.float32)

    # `preprocess_text_embeds` (condition_proj + the token refiner) is what
    # `extra_conds` runs once per sampling run, so record BOTH the raw states and
    # the refined ones: the Zig side has to reproduce each half separately.
    # inference_mode, not no_grad: the fused RMSNorm+rope kernels are in-place
    # and refuse outside inference mode ('in-place RoPE operations are
    # inference-only'), which no_grad alone does not satisfy.
    with torch.inference_mode():
        refined = model.preprocess_text_embeds(text)

    # model_base hands the model `sigma * 1000`.
    timestep = torch.tensor([SIGMA * 1000.0], dtype=torch.float32)

    layout = PackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T)
    payload = {"layout": layout, "seed": 0, "audio_scale": 1.0}

    with torch.inference_mode():
        # _forward, not forward: the audio carry is the SAMPLER's business and
        # `audio_scale == 1.0` makes forward a passthrough anyway. Calling
        # _forward keeps this fixture about the network.
        out = model._forward([video, audio], timestep, refined, minimax_payload=payload)
    out_video, out_audio = out[0], out[1]

    assert out_video.shape == video.shape, (out_video.shape, video.shape)
    assert out_audio.shape == audio.shape, (out_audio.shape, audio.shape)

    # A fixture whose output is all one sign, or tiny, or constant, cannot
    # distinguish much. Check it has structure before shipping it.
    for nm, t in (("video", out_video), ("audio", out_audio)):
        assert torch.isfinite(t).all(), nm
        assert t.std() > 1e-4, (nm, float(t.std()))
        assert (t > 0).any() and (t < 0).any(), nm

    # A corpus with no teeth passes whatever it is compared against. Re-run with
    # the swiglu halves SWAPPED and require the output to move by far more than
    # the Zig test's threshold; this is the check the first version of this
    # fixture would have failed.
    import comfy.ops as _ops
    orig = _ops._swiglu_eager

    def swapped(x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(up).mul_(gate)

    _ops.INPUT_ACT_EAGER["swiglu"] = swapped
    try:
        with torch.inference_mode():
            alt = model._forward([video, audio], timestep, refined, minimax_payload=payload)
    finally:
        _ops.INPUT_ACT_EAGER["swiglu"] = orig
    rel = (alt[0] - out_video).norm() / out_video.norm()
    assert rel > 1e-2, "the swiglu half order is indistinguishable in this fixture (rel %.2e)" % rel
    print("  swiglu-order sensitivity: rel %.4f" % rel)

    tensors = dict(sd)
    tensors["in.video"] = video
    tensors["in.audio"] = audio
    tensors["in.text"] = text
    tensors["in.refined"] = refined
    tensors["out.video"] = out_video
    tensors["out.audio"] = out_audio

    meta = {
        "config": json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in CFG.items()}),
        "shape": json.dumps({
            "text_len": TEXT_LEN, "latent_t": LATENT_T, "latent_h": LATENT_H,
            "latent_w": LATENT_W, "audio_t": AUDIO_T,
        }),
        "sigma": repr(SIGMA),
        "note": "generated by tools/gen_minimax_h3_forward.py from "
                "comfy.ldm.minimax.model.MiniMaxH3Model; do not hand-edit",
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    safetensors.torch.save_file(tensors, OUT, metadata=meta)
    print("wrote %s: %d tensors, %d bytes" % (OUT, len(tensors), os.path.getsize(OUT)))
    print("  out.video std %.5f  out.audio std %.5f"
          % (float(out_video.std()), float(out_audio.std())))


if __name__ == "__main__":
    sys.exit(main())
