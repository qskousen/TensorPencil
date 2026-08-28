#!/usr/bin/env python3
"""Pin MiniMax H3's denoise-mask handling: how a `[T][H][W]` mask becomes a
per-row TIMESTEP, and when it collapses back into a segment-level one.

The mask does not gate the arithmetic; it relabels rows on the time axis. A row
with mask value `m` runs at sigma `m * sigma_stream`, so `m = 1` is the ordinary
stream timestep and `m = 0` pins the row at the condition timestep -- the model is
told those rows are already clean. Everything that is silent when wrong:

  - ⚠️ **the 2x2 patch reduction is `amax`, not a mean.** A patch regenerates if
    ANY of its four latent cells does. A mean turns a half-covered patch into a
    half-noised one, which is a plausible image and not the reference's.
  - the mask is REPLICATE-padded up to the latent grid first, so a mask that stops
    short of an odd latent dimension extends its edge rather than reading as zero
    (i.e. as "preserve", the opposite of what a truncated mask means).
  - `rows_t = clamp(1 - m * sigma, max=t_pin)` where `t_pin` is the stream's
    condition timestep (`max(t_stream, 0.999)` for video, `max(t_stream, 1.0)` for
    audio). Note it is a MAX inside and a clamp-MAX outside.
  - ⚠️ **the video and audio streams use DIFFERENT sigmas.** Video takes the
    sampler's sigma directly; audio takes `1 - t_a`, i.e. its own shifted one. Using
    the video sigma for both is exactly right at shift parity and wrong otherwise.
  - ⚠️ **an all-ones mask returns None**, and a mask whose rows all agree COLLAPSES
    into the segment's own timestep instead of becoming a per-row table. Both are
    observable: they change the set of distinct labels, hence the modulation table's
    row count and every row index into it.
  - the audio mask is NOT patched: its rows are the mask reshaped, one per
    `[2][audio_t]` row of the packed layout.
  - `rows_to_mod_index` maps a row's timestep to `t_row[value] * 3 + tag`, the same
    (timestep, modality) interleave the modulation table uses everywhere else.

Emits src/models/assets/minimax_h3_mask.safetensors.

Usage (bounded, though this one loads no model):
    systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_minimax_h3_mask.py
"""

import json
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "minimax_h3_mask.safetensors",
)

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

from comfy.ldm.minimax.model import (  # noqa: E402
    mask_row_values, VISUAL_COND_TIMESTEP, AUDIO_COND_TIMESTEP,
)

LATENT_T, LAT_H, LAT_W = 3, 6, 8
AUDIO_T = 5
SIGMA_V = 0.6
# The two shifts H3 ships with, so the audio sigma is genuinely a different number.
SHIFT_V, SHIFT_A = 12.0, 3.0


from comfy.ldm.minimax.model import time_shift_sigma  # noqa: E402


def main():
    torch.manual_seed(20260828)

    tensors = {}
    meta = {}

    # --- the cases ---------------------------------------------------------
    # 1. all ones: the mask is inert and must reduce to None.
    # 2. binary temporal: preserve latent frame 0, generate the rest. THE case --
    #    video continuation -- and it adds no new timestep label, since m=1 gives
    #    the stream label and m=0 gives exactly the condition one.
    # 3. binary spatial with a HALF-COVERED 2x2 patch, so amax and mean disagree.
    # 4. graded: four distinct levels, which really do add labels.
    # 5. uniform but not 1: every row agrees, so it must COLLAPSE to a segment
    #    timestep rather than a per-row table.
    cases = {}

    m1 = torch.ones(LATENT_T, LAT_H, LAT_W)
    cases["all_ones"] = m1

    m2 = torch.ones(LATENT_T, LAT_H, LAT_W)
    m2[0] = 0.0
    cases["temporal"] = m2

    m3 = torch.zeros(LATENT_T, LAT_H, LAT_W)
    # a single latent COLUMN, so every 2x2 patch it touches is half covered
    m3[:, :, 3] = 1.0
    cases["spatial_half_patch"] = m3

    m4 = torch.zeros(LATENT_T, LAT_H, LAT_W)
    for i, v in enumerate((0.0, 0.25, 0.5, 1.0)):
        m4[:, :, 2 * i:2 * i + 2] = v
    cases["graded"] = m4

    m5 = torch.full((LATENT_T, LAT_H, LAT_W), 0.5)
    cases["uniform_half"] = m5

    # 6. a mask SMALLER than the latent grid, so the replicate padding runs.
    #    ⚠️ It must be short by TWO, not one: with `amax` over a 2x2 patch, a
    #    one-cell pad replicates a value that is already inside the same patch, so
    #    the pad MODE cannot change the answer. Short by two puts a whole patch row
    #    (and column) inside the padding, where replicate extends the edge and a
    #    zero pad would read as "preserve" -- the opposite of what a short mask
    #    means. Shaped 4x6 against a 6x8 latent.
    small_h, small_w = LAT_H - 2, LAT_W - 2
    m6 = torch.zeros(LATENT_T, small_h, small_w)
    m6[:, -1, :] = 1.0          # bottom row generates -> extends down
    m6[:, :, -1] = 1.0          # right column generates -> extends right
    cases["short_replicate"] = m6

    t_v = 1.0 - SIGMA_V
    sigma_a = time_shift_sigma(SIGMA_V, SHIFT_V, SHIFT_A)
    t_a = 1.0 - sigma_a
    t_pin_v = max(t_v, VISUAL_COND_TIMESTEP)
    t_pin_a = max(t_a, AUDIO_COND_TIMESTEP)
    print("sigma_v %.4f -> t_v %.4f ; sigma_a %.6f -> t_a %.6f" % (SIGMA_V, t_v, sigma_a, t_a))

    for name, mask in cases.items():
        rows = mask_row_values(mask, LATENT_T, LAT_H, LAT_W)
        tensors["mask." + name] = mask.contiguous()
        if rows is None:
            meta[name] = dict(reduced=False, mask_h=int(mask.shape[-2]), mask_w=int(mask.shape[-1]))
            print("  %-18s -> None (inert)" % name)
            tensors["rows." + name] = torch.zeros(0)
            tensors["t." + name] = torch.zeros(0)
            continue
        rows_t = (1.0 - rows * SIGMA_V).clamp(max=t_pin_v)
        uniq = rows_t.unique()
        meta[name] = dict(reduced=True, collapses=bool(uniq.numel() == 1),
                          mask_h=int(mask.shape[-2]), mask_w=int(mask.shape[-1]),
                          levels=[float(v) for v in uniq.tolist()])
        tensors["rows." + name] = rows.contiguous()
        tensors["t." + name] = rows_t.contiguous()
        print("  %-18s -> %d rows, %d distinct timestep(s)%s"
              % (name, rows.numel(), uniq.numel(), "  COLLAPSES" if uniq.numel() == 1 else ""))

    # --- the audio side ----------------------------------------------------
    # No patching at all, and its OWN sigma.
    a_mask = torch.ones(AUDIO_T * 2)
    a_mask[:4] = 0.0
    a_rows_t = (1.0 - a_mask * sigma_a).clamp(max=t_pin_a)
    tensors["audio.mask"] = a_mask.contiguous()
    tensors["audio.t"] = a_rows_t.contiguous()
    print("  audio (own sigma %.6f) -> %d rows, %d distinct"
          % (sigma_a, a_rows_t.numel(), a_rows_t.unique().numel()))

    # --- teeth -------------------------------------------------------------
    def rel(a, b):
        return float((a - b).norm() / max(float(b.norm()), 1e-12))

    # 1. amax vs mean on the half-covered patch. They MUST disagree.
    m = torch.nn.functional.pad(cases["spatial_half_patch"], (0, 0, 0, 0), mode="replicate")
    grid = m.reshape(LATENT_T, LAT_H // 2, 2, LAT_W // 2, 2)
    by_max = grid.amax(dim=(2, 4)).reshape(-1)
    by_mean = grid.mean(dim=(2, 4)).reshape(-1)
    d_reduce = float((by_mean - by_max).norm() / by_max.norm())
    assert d_reduce > 0.1, "amax and mean agree on this mask (%.3f); it has no half patches" % d_reduce
    assert torch.equal(by_max, tensors["rows.spatial_half_patch"]), "the re-run did not reproduce the reduction"

    # 2. Replicate vs ZERO padding on the short mask. Must disagree, or the pad
    #    mode is untested and a zero pad silently preserves the extended edge.
    sm = cases["short_replicate"]
    rep = mask_row_values(sm, LATENT_T, LAT_H, LAT_W)
    zero = torch.nn.functional.pad(sm, (0, LAT_W - sm.shape[-1], 0, LAT_H - sm.shape[-2]))
    zero = zero.reshape(LATENT_T, LAT_H // 2, 2, LAT_W // 2, 2).amax(dim=(2, 4)).reshape(-1)
    d_pad = rel(zero, rep)
    assert d_pad > 1e-3, "replicate and zero padding agree on this mask (%.2e)" % d_pad

    # 3. The video sigma used for the audio stream. Must differ at the real shifts.
    wrong_a = (1.0 - a_mask * SIGMA_V).clamp(max=t_pin_a)
    d_sigma = float((wrong_a - a_rows_t).norm() / a_rows_t.norm())
    assert d_sigma > 1e-3, "the two stream sigmas are indistinguishable (%.2e)" % d_sigma

    # 3. The binary temporal mask really adds no new label beyond the stream's own
    #    and the condition one -- which is what makes video continuation free.
    tt = tensors["t.temporal"].unique().tolist()
    assert len(tt) == 2, tt
    assert abs(min(tt) - t_v) < 1e-6, (min(tt), t_v)
    assert abs(max(tt) - t_pin_v) < 1e-6, (max(tt), t_pin_v)

    # 4. The graded mask DOES add labels, so the cap on distinct labels is a real
    #    constraint and not a theoretical one.
    assert len(tensors["t.graded"].unique().tolist()) == 4

    m = {
        "config": json.dumps(dict(latent_t=LATENT_T, lat_h=LAT_H, lat_w=LAT_W,
                                  audio_t=AUDIO_T, sigma_v=SIGMA_V,
                                  shift_video=SHIFT_V, shift_audio=SHIFT_A,
                                  sigma_a=sigma_a, t_v=t_v, t_a=t_a,
                                  t_pin_v=t_pin_v, t_pin_a=t_pin_a,
                                  visual_cond=VISUAL_COND_TIMESTEP,
                                  audio_cond=AUDIO_COND_TIMESTEP)),
        "cases": json.dumps(meta),
        "note": "generated by tools/gen_minimax_h3_mask.py from "
                "comfy.ldm.minimax.model.mask_row_values + the forward's row-timestep "
                "construction; do not hand-edit",
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    safetensors.torch.save_file(tensors, OUT, metadata=m)
    print("wrote %s: %d tensors, %.4f MB" % (OUT, len(tensors), os.path.getsize(OUT) / 1048576))
    print("  sensitivity: amax vs mean %.4f  pad mode %.4f  stream sigma %.4f"
          % (d_reduce, d_pad, d_sigma))


if __name__ == "__main__":
    sys.exit(main())
