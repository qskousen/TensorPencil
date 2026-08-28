#!/usr/bin/env python3
"""Pin MiniMax H3's audio VAE ENCODE by executing ComfyUI's own encoder.

The decode side is a BigVGAN vocoder; the encode side is a DAC-style
convolutional down-sampler followed by a CAUSAL-ATTENTION posterior head, so
nothing in the existing audio fixture covers it. Conventions that are silent when
wrong:

  - ⚠️ **`Snake1d`, not `SnakeBeta`.** The encoder's activation stores alpha
    LINEAR and uses the SAME parameter as beta; the decoder's stores alpha and
    beta separately and in LOG scale. Same family, three differences.
  - ⚠️ **The residual dilations are 1, 3, 9** in the encoder and 1, 3, 5 in the
    decoder. Reusing the decoder's cycle is finite and wrong.
  - a `ResidualUnit` is Snake -> Conv(k=7, dilated) -> Snake -> Conv(k=1), and the
    residual add CENTER-CROPS `x` when the conv shortened it.
  - an `EncoderBlock`'s three residual units run at `dim // 2`, i.e. the PREVIOUS
    stage's width, and only the final strided conv widens to `dim`. Its kernel is
    `2 * stride` with padding `ceil(stride / 2)`.
  - ⚠️ **The posterior head's attention is CAUSAL**, then takes the MEAN OVER
    HEADS, then `adaptive_avg_pool1d` along the FEATURE axis down to the latent
    width. Pooling the time axis instead is the obvious misreading and gives a
    plausible shape.
  - ⚠️ **`zero_k_bias` is a zero BUFFER, not a parameter**: the fused qkv has no
    bias of its own and the k third of the concatenated bias is zeros.
  - `AttnProjection` is `proj(norm3(x)) + attn(norm1(x))`, then `+ mlp(norm2(x))`
    -- note norm3 feeds the projection and norm1 the attention, which is easy to
    swap.
  - `GeGluMlp` is `w2(gelu_tanh(w0(x)) * w1(x))`, gelu-TANH.
  - encode right-pads the waveform to a multiple of the hop, keeps the posterior
    MEAN (`mean_proj`, never `logs_proj`, and no sampling), and normalizes per
    channel.
  - the two stereo channels fold into the BATCH and are encoded independently.

Emits src/models/assets/minimax_h3_audio_encode.safetensors.

Usage (BOUND IT):
    systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_minimax_h3_audio_encode.py
"""

import json
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "minimax_h3_audio_encode.safetensors",
)

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.ldm.minimax.audio_vae as av  # noqa: E402

# Two strides so the widening schedule runs, and both are the real KINDS: a 2
# (even kernel 4, padding 1) and a 5 (kernel 10, padding 3), which are the two
# padding relations in the real (2,4,4,5,5).
ENC_DIM = 8
ENC_RATES = (2, 5)
LATENT_DIM = 64
# 6, not 8, so the posterior head's `adaptive_avg_pool1d` runs with UNEVEN bins
# (head_dim 16 -> 6). The real model pools 256 -> 32, which divides evenly and
# hides the general start/end formula behind a plain block mean.
VAE_CH = 6
HEADS = 4
# Enough samples that the strided convs all have something to chew, and NOT a
# multiple of the hop, so the right-pad actually happens.
SAMPLES = 137


def main():
    torch.manual_seed(20260828)
    torch.set_num_threads(4)

    class NoDecoder(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()

    # The BigVGAN decoder is built at its real size whatever the encoder config
    # is, and encode never touches it; the decode side has its own fixture.
    real_bigvgan = av.BigVGAN
    av.BigVGAN = NoDecoder
    try:
        vae = av.MiniMaxH3AudioVAE(
            encoder_dim=ENC_DIM, encoder_rates=ENC_RATES, latent_dim=LATENT_DIM,
            vae_latent_channels=VAE_CH,
        )
    finally:
        av.BigVGAN = real_bigvgan
    assert (LATENT_DIM // HEADS) % VAE_CH != 0, \
        "head_dim divides the latent width evenly, so the adaptive pooling is untested"
    vae.pre_block = av.AttnProjection(LATENT_DIM, VAE_CH, num_heads=HEADS)
    vae.eval()
    n = sum(p.numel() for p in vae.parameters())
    print("toy audio encoder: %d parameters (%.2f MB f32)" % (n, n * 4 / 1048576))
    assert n < 5_000_000, "the 'toy' encoder is not toy: %d parameters" % n

    sd = {}
    for name, p in vae.state_dict().items():
        if name in ("latents_mean", "latents_std"):
            sd[name] = torch.randn(p.shape, dtype=torch.float32) * 0.3 if "mean" in name \
                else 1.0 + torch.rand(p.shape, dtype=torch.float32)
        elif "zero_k_bias" in name:
            sd[name] = torch.zeros(p.shape, dtype=torch.float32)
        elif "alpha" in name or "beta" in name:
            # Snake1d's alpha is LINEAR and doubles as beta, so it must stay well
            # away from zero or the activation blows up.
            sd[name] = 0.5 + torch.rand(p.shape, dtype=torch.float32)
        elif p.dim() == 3:
            fan_in = p.shape[1] * p.shape[2]
            sd[name] = torch.randn(p.shape, dtype=torch.float32) / (fan_in ** 0.5)
        elif p.dim() == 2:
            sd[name] = torch.randn(p.shape, dtype=torch.float32) / (p.shape[1] ** 0.5)
        elif name.endswith(".weight"):
            sd[name] = 1.0 + torch.randn(p.shape, dtype=torch.float32) * 0.1
        else:
            sd[name] = torch.randn(p.shape, dtype=torch.float32) * 0.05
    vae.load_state_dict(sd, strict=True)
    for prm in vae.parameters():
        prm.requires_grad_(False)

    hop = 1
    for r in ENC_RATES:
        hop *= r
    # Stereo, and the two channels must DIFFER or a port that encoded one and
    # copied it would pass.
    wav = torch.rand(1, 2, SAMPLES, dtype=torch.float32) * 2.0 - 1.0
    wav[0, 1] = torch.rand(SAMPLES) * 2.0 - 1.0

    with torch.inference_mode():
        z = vae.encode(wav)
    exp_t = -(-SAMPLES // hop)
    assert z.shape == (1, VAE_CH, 2, exp_t), (z.shape, exp_t)
    assert torch.isfinite(z).all()
    assert z.std() > 1e-3, float(z.std())
    print("  waveform [1,2,%d] hop %d -> latent %s std %.4f"
          % (SAMPLES, hop, list(z.shape), float(z.std())))

    # Names are the checkpoint's own, unprefixed, so the loader here is the one the
    # real file exercises.
    tensors = {}
    for k, v in sd.items():
        if k.startswith("decoder.") or k.startswith("dec_in_proj."):
            continue
        tensors[k] = v
    tensors["in.waveform"] = wav
    tensors["out.latent"] = z

    # --- teeth -------------------------------------------------------------
    def rel(a, b):
        return float((a - b).norm() / b.norm())

    # 1. The two stereo channels really differ.
    ch_rel = rel(z[0, :, 0], z[0, :, 1])
    assert ch_rel > 0.1, "the stereo channels are too similar (%.3f)" % ch_rel

    # 2. Snake1d sharing alpha as beta, versus treating alpha as LOG scale the way
    #    the decoder's SnakeBeta does. That substitution must move the answer.
    orig = av.Snake1d.forward
    av.Snake1d.forward = lambda self, x: av.snake(
        x, torch.exp(av.comfy.ops.cast_to_input(self.alpha, x)),
        torch.exp(av.comfy.ops.cast_to_input(self.alpha, x)))
    try:
        with torch.inference_mode():
            alt = vae.encode(wav)
    finally:
        av.Snake1d.forward = orig
    d_snake = rel(alt, z)
    assert d_snake > 1e-2, "linear vs log alpha is indistinguishable (%.2e)" % d_snake

    # 3. A NON-causal attention in the posterior head.
    orig_attn = av.CausalAttention.forward

    def noncausal(self, x):
        B, N, C = x.shape
        qkv = F.linear(x, weight=self.qkv.weight,
                       bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = F.adaptive_avg_pool1d(torch.mean(y, dim=1), self.out_dim)
        return self.proj(y)

    av.CausalAttention.forward = noncausal
    try:
        with torch.inference_mode():
            alt = vae.encode(wav)
    finally:
        av.CausalAttention.forward = orig_attn
    d_causal = rel(alt, z)
    assert d_causal > 1e-3, "causal vs full attention is indistinguishable (%.2e)" % d_causal

    # 4. Taking `logs_proj` instead of `mean_proj`.
    with torch.inference_mode():
        b, s, length = wav.shape
        rp = -(-length // hop) * hop - length
        w2 = F.pad(wav, (0, rp)).reshape(b * s, 1, -1)
        h = vae.encoder(w2)
        h = vae.pre_block(h.transpose(1, 2)).transpose(1, 2)
        wrong = vae.logs_proj(h)
        m = vae.latents_mean.view(1, -1, 1)
        sd_ = vae.latents_std.view(1, -1, 1)
        wrong = ((wrong - m) / sd_).reshape(b, s, VAE_CH, -1).permute(0, 2, 1, 3)
    d_head = rel(wrong, z)
    assert d_head > 1e-2, "mean_proj vs logs_proj is indistinguishable (%.2e)" % d_head

    # 5. norm1 (into the attention) and norm3 (into the projection) swapped.
    orig_ap = av.AttnProjection.forward

    def swapped(self, x):
        x = self.proj(self.norm1(x)).add_(self.attn(self.norm3(x)))
        return x.add_(self.mlp(self.norm2(x)))

    av.AttnProjection.forward = swapped
    try:
        with torch.inference_mode():
            alt = vae.encode(wav)
    finally:
        av.AttnProjection.forward = orig_ap
    d_norms = rel(alt, z)
    assert d_norms > 1e-3, "norm1/norm3 are indistinguishable (%.2e)" % d_norms

    meta = {
        "config": json.dumps(dict(encoder_dim=ENC_DIM, encoder_rates=list(ENC_RATES),
                                  latent_dim=LATENT_DIM, vae_latent_channels=VAE_CH,
                                  heads=HEADS, hop=hop, samples=SAMPLES,
                                  latent_t=exp_t)),
        "note": "generated by tools/gen_minimax_h3_audio_encode.py from "
                "comfy.ldm.minimax.audio_vae.MiniMaxH3AudioVAE.encode; do not hand-edit",
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    safetensors.torch.save_file(tensors, OUT, metadata=meta)
    print("wrote %s: %d tensors, %.3f MB" % (OUT, len(tensors), os.path.getsize(OUT) / 1048576))
    print("  sensitivity: snake scale %.4f  causal %.4f  mean-head %.4f  norms %.4f  stereo %.3f"
          % (d_snake, d_causal, d_head, d_norms, ch_rel))


if __name__ == "__main__":
    sys.exit(main())
