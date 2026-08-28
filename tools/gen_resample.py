#!/usr/bin/env python3
"""Pin torchaudio's `functional.resample` at its defaults, which is how every
reference soundtrack reaches the audio VAE's 32 kHz.

Anything that merely "resamples" will produce plausible audio; only matching the
reference's filter reproduces the reference's latent. The defaults are
`lowpass_filter_width=6`, `rolloff=0.99`, `resampling_method="sinc_interp_hann"`,
and the pieces that are silent when wrong:

  - both rates are divided by their GCD FIRST, and every constant below is derived
    from the reduced pair. 48000 -> 32000 reduces to 3 -> 2, so the filter is 23
    taps; using the raw rates gives a 16001-tap filter and the same answer far
    more slowly, but 44100 -> 32000 reduces to 441 -> 320 and the reduction is
    what makes the polyphase structure exist at all.
  - `base_freq = min(orig, new) * rolloff`, and the ROLLOFF is inside the window
    argument, the sinc argument AND the output scale.
  - the window is `cos(t * pi / width / 2) ** 2` evaluated at the SAME clamped `t`
    the sinc uses, not on a regular grid: `t` is clamped to +/- lowpass_filter_width
    BEFORE both.
  - the signal is zero-padded `width` on the left and `width + orig_freq` on the
    right, convolved with stride `orig_freq`, and the result is TRANSPOSED before
    flattening (phase-major, not time-major) -- flattening the other way
    interleaves the polyphase branches and sounds like a wrong pitch.
  - the output is then truncated to `ceil(new_freq * length / orig_freq)`.

Emits src/core/assets/resample.safetensors: the kernels for three rate pairs plus
a resampled signal for each.

Usage (bounded, though this one is tiny):
    systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_resample.py
"""

import json
import math
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "core", "assets", "resample.safetensors",
)

sys.path.insert(0, "/home/qt/genai/comfyui")

import torch  # noqa: E402
import safetensors.torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402
from torchaudio.functional.functional import _get_sinc_resample_kernel  # noqa: E402

# The three rate relations that matter: a large-GCD downsample (the easy one), the
# awkward 44.1k -> 32k that reduces to 441 -> 320, and an UPSAMPLE, where the
# rolloff exists to stop edge artifacts rather than to antialias.
CASES = [(48000, 32000), (44100, 32000), (16000, 32000)]
LENGTH = 900


def main():
    torch.manual_seed(20260828)
    torch.set_num_threads(4)

    tensors = {}
    meta = {}
    # A signal with content across the band, so a filter that is merely
    # approximately right shows up. Pure noise would too, but a chirp makes the
    # failure mode legible if anyone ever plots it.
    n = torch.arange(LENGTH, dtype=torch.float32)
    sig = (torch.sin(2 * math.pi * (0.002 + 0.0004 * n) * n)
           + 0.3 * torch.rand(LENGTH) - 0.15)

    for orig, new in CASES:
        gcd = math.gcd(orig, new)
        kernel, width = _get_sinc_resample_kernel(
            orig, new, gcd, 6, 0.99, "sinc_interp_hann", None,
            torch.device("cpu"), torch.float32)
        # The SAME filter in f64. The reference applies the f32 one, but it is the
        # f64 one that the formula defines, and 441 -> 320 has 320 phases whose
        # near-zero taps lose most of their significance in f32: the two differ by
        # 1e-5 relative. An implementation that computes in f64 is therefore
        # CLOSER to the filter than the reference is, and comparing it against the
        # f32 kernel would read as a 1e-5 error with nothing wrong. Both go in the
        # file so the test can state which is which.
        kernel64, _ = _get_sinc_resample_kernel(
            orig, new, gcd, 6, 0.99, "sinc_interp_hann", None,
            torch.device("cpu"), torch.float64)
        out = AF.resample(sig, orig, new)
        exp = math.ceil((new // gcd) * LENGTH / (orig // gcd))
        assert out.shape[0] == exp, (out.shape, exp)
        assert torch.isfinite(out).all()

        tag = "%d_%d" % (orig, new)
        # kernel is [new/gcd, 1, 2*width + orig/gcd]; drop the singleton in-channel
        tensors["kernel." + tag] = kernel[:, 0].contiguous()
        tensors["kernel64." + tag] = kernel64[:, 0].float().contiguous()
        tensors["out." + tag] = out
        f32_floor = float((kernel64.float() - kernel).norm() / kernel.norm())
        meta[tag] = dict(orig=orig, new=new, gcd=gcd, width=int(width),
                         phases=int(kernel.shape[0]), taps=int(kernel.shape[2]),
                         out_len=int(out.shape[0]), f32_floor=f32_floor)
        print("  %6d -> %6d  gcd %5d  width %2d  %3d phases x %4d taps  out %d"
              "  (the reference's own f32 floor: %.2e)"
              % (orig, new, gcd, width, kernel.shape[0], kernel.shape[2],
                 out.shape[0], f32_floor))
    tensors["in.signal"] = sig

    # --- teeth -------------------------------------------------------------
    def rel(a, b):
        return float((a - b).norm() / b.norm())

    # 1. A pass-through must be exactly the identity, or the padding/truncation
    #    bookkeeping is off by a sample somewhere.
    same = AF.resample(sig, 32000, 32000)
    assert torch.equal(same, sig), "resampling to the same rate is not the identity"

    # 2. The rolloff must matter. 1.0 (no rolloff) against the default 0.99.
    k99, _ = _get_sinc_resample_kernel(44100, 32000, 100, 6, 0.99, "sinc_interp_hann",
                                       None, torch.device("cpu"), torch.float32)
    k10, _ = _get_sinc_resample_kernel(44100, 32000, 100, 6, 1.0, "sinc_interp_hann",
                                       None, torch.device("cpu"), torch.float32)
    d_roll = rel(k10, k99)
    assert d_roll > 1e-3, "the rolloff is indistinguishable (%.2e)" % d_roll

    # 3. The Hann window against the Kaiser one, the other method the same code
    #    path offers -- so a port that dropped the window entirely, or picked the
    #    other, cannot pass.
    kk, _ = _get_sinc_resample_kernel(44100, 32000, 100, 6, 0.99, "sinc_interp_kaiser",
                                      None, torch.device("cpu"), torch.float32)
    d_win = rel(kk, k99)
    assert d_win > 1e-2, "the window choice is indistinguishable (%.2e)" % d_win

    # 4. Phase-major vs time-major flattening. The polyphase output really is
    #    transposed before the reshape, and the two orders must disagree.
    orig_r, new_r = 441, 320
    padded = torch.nn.functional.pad(sig[None], (int(meta["44100_32000"]["width"]),
                                                 int(meta["44100_32000"]["width"]) + orig_r))
    conv = torch.nn.functional.conv1d(padded[:, None], k99, stride=orig_r)
    right = conv.transpose(1, 2).reshape(1, -1)[0, :meta["44100_32000"]["out_len"]]
    wrong = conv.reshape(1, -1)[0, :meta["44100_32000"]["out_len"]]
    assert rel(right, tensors["out.44100_32000"]) < 1e-6, "the re-run did not reproduce the output"
    d_order = rel(wrong, right)
    assert d_order > 0.1, "the two flatten orders agree (%.2e)" % d_order

    m = {
        "config": json.dumps(dict(lowpass_filter_width=6, rolloff=0.99,
                                  method="sinc_interp_hann", length=LENGTH)),
        "cases": json.dumps(meta),
        "note": "generated by tools/gen_resample.py from "
                "torchaudio.functional.resample; do not hand-edit",
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    safetensors.torch.save_file(tensors, OUT, metadata=m)
    print("wrote %s: %d tensors, %.3f MB" % (OUT, len(tensors), os.path.getsize(OUT) / 1048576))
    print("  sensitivity: rolloff %.4f  window %.4f  flatten order %.4f"
          % (d_roll, d_win, d_order))


if __name__ == "__main__":
    sys.exit(main())
