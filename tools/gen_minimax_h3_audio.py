#!/usr/bin/env python3
"""Pin MiniMax H3's audio VAE DECODE by executing ComfyUI's own BigVGAN.

Same approach as the DiT and video-VAE fixtures. What this pins is the 1-D op
family nothing else in the engine has:

  - SnakeBeta with alpha/beta in LOG scale
  - the ANTI-ALIASED activation (upsample x2 -> snake -> downsample x2) with
    kaiser-sinc filters, replicate padding, and its asymmetric slice constants
  - dilated `convs1` (1, 3, 5) paired with undilated `convs2`, and the six
    distinct activations pairing as [::2] / [1::2]
  - the three resblocks per stage SUMMED and averaged, not chained
  - ConvTranspose1d upsampling with `padding=(k-u)//2`
  - the final clamp to [-1, 1], no tanh, no bias

Kernel sizes are the real ones where they are load-bearing: the kaiser filters
are 12 taps and the resblock kernels/dilations are the reference's, because the
padding constants are derived from them.

Emits src/models/assets/minimax_h3_audio.safetensors.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_minimax_h3_audio.py
"""

import json
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "minimax_h3_audio.safetensors",
)

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.ldm.minimax.audio_vae as av  # noqa: E402

# Two stages, one with rate 5 (kernel 9, the odd pairing) and one with rate 2
# (kernel 4), so BOTH kernel/rate relations are exercised. Channels are tiny but
# must halve cleanly per stage.
NUM_MELS = 8
INITIAL_CH = 16
UP_RATES = (5, 2)
UP_KERNELS = (9, 4)
RES_KERNELS = (3, 7, 11)
RES_DILATIONS = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
LATENT_CH = 32
LAT_T = 3


def main():
    torch.manual_seed(20260824)

    dec = av.BigVGAN(
        num_mels=NUM_MELS,
        upsample_initial_channel=INITIAL_CH,
        upsample_rates=UP_RATES,
        upsample_kernel_sizes=UP_KERNELS,
        resblock_kernel_sizes=RES_KERNELS,
        resblock_dilation_sizes=RES_DILATIONS,
    )
    dec.eval()

    sd = {}
    for name, p in dec.state_dict().items():
        if name.endswith(".filter"):
            # the kaiser-sinc filters are DERIVED, not trained: keep what the
            # module computed, since the Zig side loads them from the checkpoint
            t = p.clone()
        elif ".act.alpha" in name or ".act.beta" in name:
            # log scale: exp() of these must stay O(1) or the snake saturates
            t = torch.randn(p.shape, dtype=torch.float32) * 0.3
        elif name.endswith(".bias"):
            t = torch.randn(p.shape, dtype=torch.float32) * 0.1
        elif p.dim() == 3:
            # fan-in over (in_ch * k), so activations stay O(1) through the stack
            fan_in = p.shape[1] * p.shape[2]
            t = torch.randn(p.shape, dtype=torch.float32) / (fan_in ** 0.5)
        else:
            t = torch.randn(p.shape, dtype=torch.float32) * 0.1
        sd[name] = t
    dec.load_state_dict(sd, strict=True)
    for prm in dec.parameters():
        prm.requires_grad_(False)

    # `dec_in_proj` and the latent statistics live on the VAE wrapper.
    dec_in_w = torch.randn(NUM_MELS, LATENT_CH, 1, dtype=torch.float32) / (LATENT_CH ** 0.5)
    dec_in_b = torch.randn(NUM_MELS, dtype=torch.float32) * 0.1
    latents_mean = torch.randn(LATENT_CH, dtype=torch.float32) * 0.3
    latents_std = 1.0 + torch.rand(LATENT_CH, dtype=torch.float32)

    z = torch.randn(1, LATENT_CH, 2, LAT_T, dtype=torch.float32)

    def run():
        with torch.inference_mode():
            # the wrapper's decode: [B,C,S,T] -> [B*S,C,T], denormalize, project,
            # vocode, then reshape back to [B, S, L]
            b, c, s, t = z.shape
            zz = z.permute(0, 2, 1, 3).reshape(b * s, c, t)
            zz = zz * latents_std.view(1, -1, 1) + latents_mean.view(1, -1, 1)
            x = torch.nn.functional.conv1d(zz, dec_in_w, dec_in_b)
            return dec(x).reshape(b, s, -1)

    out = run()
    total_up = 1
    for r in UP_RATES:
        total_up *= r
    assert out.shape == (1, 2, LAT_T * total_up), (out.shape, total_up)
    assert torch.isfinite(out).all()
    assert out.std() > 1e-3, float(out.std())
    interior = float(((out > -1 + 1e-4) & (out < 1 - 1e-4)).float().mean())
    assert interior > 0.5, "output is mostly clamped (interior %.3f)" % interior
    # The two stereo channels must DIFFER: they share the vocoder, so an
    # implementation that decoded one and copied it would pass everything else.
    chan_rel = float((out[0, 0] - out[0, 1]).norm() / out[0, 0].norm())
    assert chan_rel > 0.1, "the stereo channels are too similar (rel %.3f)" % chan_rel

    # Teeth: a plain pointwise snake instead of the anti-aliased one is the
    # obvious wrong implementation, and it must move the output a lot.
    orig = av.Activation1d.forward
    av.Activation1d.forward = lambda self, x: self.act(x)
    try:
        alt = run()
    finally:
        av.Activation1d.forward = orig
    aa_rel = float((alt - out).norm() / out.norm())
    assert aa_rel > 1e-2, "anti-aliasing is indistinguishable (rel %.2e)" % aa_rel

    # ...and so is using alpha/beta raw rather than exponentiating them.
    orig_snake = av.SnakeBeta.forward

    def raw_snake(self, x):
        a = av.comfy.ops.cast_to_input(self.alpha, x).view(1, -1, 1)
        b = av.comfy.ops.cast_to_input(self.beta, x).view(1, -1, 1)
        return av.snake(x, a, b)

    av.SnakeBeta.forward = raw_snake
    try:
        alt2 = run()
    finally:
        av.SnakeBeta.forward = orig_snake
    log_rel = float((alt2 - out).norm() / out.norm())
    assert log_rel > 1e-2, "the log-scale alpha/beta is indistinguishable (rel %.2e)" % log_rel

    tensors = {"decoder." + k: v for k, v in sd.items()}
    tensors["dec_in_proj.weight"] = dec_in_w
    tensors["dec_in_proj.bias"] = dec_in_b
    tensors["latents_mean"] = latents_mean
    tensors["latents_std"] = latents_std
    tensors["in.z"] = z
    tensors["out.audio"] = out

    meta = {
        "config": json.dumps(dict(num_mels=NUM_MELS, initial_ch=INITIAL_CH,
                                  up_rates=list(UP_RATES), up_kernels=list(UP_KERNELS),
                                  res_kernels=list(RES_KERNELS), latent_ch=LATENT_CH)),
        "shape": json.dumps(dict(t=LAT_T, samples=LAT_T * total_up)),
        "note": "generated by tools/gen_minimax_h3_audio.py from "
                "comfy.ldm.minimax.audio_vae.BigVGAN; do not hand-edit",
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    safetensors.torch.save_file(tensors, OUT, metadata=meta)
    print("wrote %s: %d tensors, %d bytes" % (OUT, len(tensors), os.path.getsize(OUT)))
    print("  out std %.5f interior %.3f  sensitivity: anti-alias %.4f log-scale %.4f  stereo %.3f"
          % (float(out.std()), interior, aa_rel, log_rel, chan_rel))


if __name__ == "__main__":
    sys.exit(main())
