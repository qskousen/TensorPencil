#!/usr/bin/env python3
"""Qwen-Image 2.1's sampling sigmas, by EXECUTING diffusers' own scheduler.

The reference here is diffusers, not ComfyUI: ComfyUI samples 2.1 on its Flux table
at one fixed mu (0.69, which is diffusers' value at 1024x1024 only) and skips the
terminal stretch. diffusers' `QwenImage21Pipeline` instead
  - takes `linspace(1, 1/N, N)`, or the release's `sample_sigmas` when it has them;
  - computes mu per render from the TARGET's latent token count, linear between
    `base_shift` at `base_image_seq_len` and `max_shift` at `max_image_seq_len`;
  - and lets `FlowMatchEulerDiscreteScheduler.set_timesteps` shift (exponential),
    stretch to `shift_terminal`, and append the terminal 0.
The two scheduler configs are the released ones, copied verbatim from
`Qwen/Qwen-Image-2.1` and `Qwen/Qwen-Image-2.1-Turbo` (scheduler_config.json), and the
turbo `sample_sigmas` from that repo's model_index.json.

Sigmas are stored as f32 BITS, so the Zig side is held to the reference exactly.

Emits src/core/assets/qwen_image21_sigmas.json.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_qwen_image21_sigmas.py
"""

import json
import os

import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "core", "assets", "qwen_image21_sigmas.json")

BASE = {
    "base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": False, "max_image_seq_len": 8192,
    "max_shift": 0.9, "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": 0.02,
    "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
    "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False,
}
TURBO = dict(BASE, shift_terminal=None, use_dynamic_shifting=False)
TURBO_SAMPLE_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


def calculate_shift(image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift):
    # diffusers.pipelines.qwenimage21.pipeline_qwenimage21.calculate_shift
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def run(config, steps, tokens, sample_sigmas):
    sched = FlowMatchEulerDiscreteScheduler(**config)
    sigmas = sample_sigmas if sample_sigmas is not None else np.linspace(1.0, 1 / steps, steps)
    mu = calculate_shift(tokens, config["base_image_seq_len"], config["max_image_seq_len"],
                         config["base_shift"], config["max_shift"])
    sched.set_timesteps(sigmas=sigmas, mu=mu)
    out = sched.sigmas.to(dtype=__import__("torch").float32).numpy()
    return [int(v) for v in out.view(np.uint32)]


cases = []
# Not 1 step: its only sigma is 1.0, and the terminal stretch then divides 0 by 0,
# so diffusers itself returns NaN there.
for steps in (2, 8, 25, 50):
    for (h, w) in ((16, 16), (32, 32), (64, 64), (96, 96), (128, 128), (48, 80), (130, 130)):
        cases.append({"kind": "base", "steps": steps, "tokens": h * w, "bits": run(BASE, steps, h * w, None)})
for (h, w) in ((32, 32), (64, 64), (128, 128)):
    cases.append({"kind": "turbo", "steps": 8, "tokens": h * w, "bits": run(TURBO, 8, h * w, TURBO_SAMPLE_SIGMAS)})

# The point of the fixture: diffusers' mu moves with the resolution where ComfyUI's
# is fixed, so a schedule that ignored the token count would pass only at 1024x1024.
a = [c for c in cases if c["kind"] == "base" and c["steps"] == 25]
assert len({tuple(c["bits"]) for c in a}) == len(a), "resolutions must give different schedules"

with open(OUT, "w") as f:
    json.dump({"turbo_sample_sigmas": TURBO_SAMPLE_SIGMAS, "cases": cases}, f)
print(f"wrote {OUT}: {len(cases)} cases")
