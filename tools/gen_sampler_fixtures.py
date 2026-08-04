#!/usr/bin/env python3
"""Reference fixtures for DPM++ 2M SDE (Heun and midpoint) — the solver AND its
Brownian-tree noise source.

Two tiers, because two very different things need pinning:

  1. **The Brownian tree** (`brownian` key). `torchsde.BrownianTree` as ComfyUI's
     `BrownianTreeNoiseSampler` constructs it, queried on a descending sigma sweep.
     This is the part that cannot be checked by reasoning: it composes numpy's
     `SeedSequence` entropy mixing, a dyadic interval tree quantised by Python's
     `round(x, 6)`, and `torch.randn` per node. A wrong seed derivation or a
     one-ulp difference in a midpoint yields perfectly plausible noise that is not
     ComfyUI's, with nothing to show for it.

  2. **Whole trajectories** (`trajectories`). ComfyUI's *actual*
     `sample_dpmpp_2m_sde` / `..._heun` driven over a toy analytic denoiser, once
     per family. The toy model is `denoised = (x + c) / (1 + sigma)` — pure f32
     arithmetic with no transcendentals, so it is reproducible exactly in Zig and
     the comparison measures the solver, not libm. Both families are covered
     because they take *different* half-logSNR branches (`CONST` uses `-logit`,
     `EPS` uses `-log`) and only the CONST one needs the first-sigma offset.

Driven with ComfyUI's own code, not a reimplementation: the whole point is that
the fixture and the implementation under test share no assumption.

Usage (ComfyUI's `nvenv`):
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_sampler_fixtures.py
"""

import json
import os
import sys

COMFY = "/home/qt/genai/comfyui"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `tp_core` is rooted at src/core/core.zig and can only `@embedFile` under its own path.
OUT = os.path.join(REPO, "src", "core", "assets", "dpmpp_sde_fixtures.json")

# Fixed everywhere: the fixture must be reproducible byte-for-byte from this script.
SEED = 20260803
# Small enough for a compact fixture, >= 16 so torch takes its `normal_fill` path
# (the one `torch_rng.zig` reproduces) and a multiple of 16 so there is no overlapping
# final block.
N = 64

sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

# ⚠️ Imported as a library ComfyUI ignores argv entirely unless this is called first;
# see `gen_sdxl_fixtures.py` for the full story. Nothing here depends on a device, but
# a silently-GPU reference is exactly the failure mode that wastes a day.
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import comfy.model_sampling as ms  # noqa: E402
import comfy.samplers  # noqa: E402
from comfy.k_diffusion import sampling as kds  # noqa: E402


def f32_list(t):
    """f32 -> JSON. float(f32) is exact and Python's repr round-trips a double, so the
    values reach Zig bit-for-bit."""
    return [float(v) for v in t.detach().flatten().to(torch.float32)]


# --- 1. The Brownian tree on its own ---------------------------------------------


def brownian_fixture():
    """`BrownianTreeNoiseSampler` over a descending sweep, plus two out-of-order
    queries that pin the halfway-tree property (the path must not depend on query
    order) and one that spans several nodes."""
    sigmas = [14.614642, 9.0, 5.0, 2.5, 1.0, 0.4, 0.1, 0.029168]
    sigma_min, sigma_max = min(sigmas), max(sigmas)
    x = torch.zeros(1, 4, 4, 4, dtype=torch.float32)  # shape only; N = 64
    assert x.numel() == N

    queries = [(sigmas[i], sigmas[i + 1]) for i in range(len(sigmas) - 1)]
    # Deliberately not in sweep order, and one interval that is not a single node.
    queries += [(9.0, 1.0), (0.4, 0.1), (14.614642, 0.029168)]

    ns = kds.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=SEED, cpu=True)
    samples = [f32_list(ns(torch.tensor(a), torch.tensor(b))) for a, b in queries]
    return dict(n=N, t0=sigma_min, t1=sigma_max, seed=SEED,
                queries=[list(q) for q in queries], samples=samples)


# --- 2. Whole trajectories through ComfyUI's own solver ---------------------------


class ToyDenoiser:
    """`denoised = (x + c) / (1 + sigma)`.

    Pure f32 add/divide: no transcendental, so Zig reproduces it exactly and any
    disagreement in the trajectory is the solver's or the noise's. It is also a
    contraction toward `c`, so the trajectory stays bounded over a full schedule
    instead of blowing up and hiding a sign error in an overflow.
    """

    def __init__(self, c, model_sampling):
        self.c = c
        # `sample_dpmpp_2m_sde` reaches for the model_sampling object through this
        # exact attribute chain, so the stub has to present it.
        class _Patcher:
            def get_model_object(self_inner, name):
                assert name == "model_sampling"
                return model_sampling
        class _Inner:
            model_patcher = _Patcher()
        self.inner_model = _Inner()

    def __call__(self, x, sigma, **kwargs):
        s = float(sigma.flatten()[0])
        return (x + self.c) / np.float32(1.0 + s)


def trajectory(name, model_sampling, sigmas, solver_type, eta, s_noise):
    torch.manual_seed(0)
    c = torch.randn(1, 4, 4, 4, dtype=torch.float32)
    x0 = torch.randn(1, 4, 4, 4, dtype=torch.float32)
    sig = torch.tensor(sigmas, dtype=torch.float32)

    model = ToyDenoiser(c, model_sampling)
    fn = kds.sample_dpmpp_2m_sde_heun if solver_type == "heun" else kds.sample_dpmpp_2m_sde
    out = fn(model, x0.clone(), sig, extra_args={"seed": SEED},
             disable=True, eta=eta, s_noise=s_noise)
    return dict(name=name, family=("const" if isinstance(model_sampling, ms.CONST) else "eps"),
                solver_type=solver_type, eta=eta, s_noise=s_noise, seed=SEED, n=N,
                sigmas=[float(v) for v in sig], c=f32_list(c), x0=f32_list(x0),
                x_out=f32_list(out))


def const_sampling(shift=1.15):
    """krea2 / flux: `ModelSamplingFlux` + `CONST`, exactly what `model_sampling()`
    composes for `ModelType.FLUX`."""
    class MS(ms.ModelSamplingFlux, ms.CONST):
        pass
    m = MS()
    m.set_parameters(shift=shift)
    return m


def eps_sampling():
    """SD1.5 / SDXL: `ModelSamplingDiscrete` + `EPS`."""
    class MS(ms.ModelSamplingDiscrete, ms.EPS):
        pass
    return MS()


def simple_schedule(steps, shift=1.15):
    """ComfyUI's "simple" scheduler over ModelSamplingFlux — what `sampler.simpleSchedule`
    reproduces. Taken from the model's own sigma table so the fixture cannot drift from
    the reference's discretization."""
    m = const_sampling(shift)
    total = len(m.sigmas)
    ss = total / steps
    out = [float(m.sigmas[-(1 + int(x * ss))]) for x in range(steps)]
    return out + [0.0]


# --- 3. Every scheduler, both sigma tables ---------------------------------------


SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple",
              "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"]
SCHED_STEPS = [4, 10, 20, 30]


def scheduler_fixture(const_ms, eps_ms):
    """ComfyUI's `calculate_sigmas` for every scheduler x both families x several step
    counts, called through ComfyUI's own dispatcher (which owns the `use_ms` split
    between `f(model_sampling, steps)` and `f(n, sigma_min, sigma_max)`).

    ⚠️ Several of these do NOT return `steps + 1` sigmas: `ddim_uniform` strides the
    table, and `beta` de-duplicates repeated indices. The count is emitted so the Zig
    side asserts it rather than assuming, and so the sampling loop is reminded to take
    its step count from the schedule.
    """
    out = {}
    for fam, ms in (("flux", const_ms), ("sd", eps_ms)):
        for name in SCHEDULERS:
            for steps in SCHED_STEPS:
                sig = comfy.samplers.calculate_sigmas(ms, name, steps)
                out[f"{fam}|{name}|{steps}"] = [float(v) for v in sig]
    return out


def table_fixture(const_ms, eps_ms):
    """The two `model_sampling.sigmas` tables' own endpoints and a few interior entries.

    ⚠️ These are computed by torch in **f32** (`ModelSamplingFlux.set_parameters` builds
    them from an f32 `arange/timesteps` tensor), so a Zig port computing the same formula
    in f64 lands a fraction of an ulp away — which is invisible to Euler and decisive for
    the Brownian tree. Pinned explicitly so the interpolation-space lesson is not
    re-learned on the flux table.
    """
    out = {}
    for fam, ms in (("flux", const_ms), ("sd", eps_ms)):
        s = ms.sigmas
        idx = [0, 1, 2, len(s) // 3, len(s) // 2, len(s) - 2, len(s) - 1]
        # ⚠️ A hash over EVERY entry's raw f32 bits, because sampling a handful of
        # indices is not a real check: the `scalar / tensor` reciprocal convention only
        # moves 1.65% of the flux table, so 7 samples would miss it with ~90%
        # probability. One number, exhaustive.
        import numpy as _np
        h = 0xcbf29ce484222325
        for byte in _np.asarray(s, dtype=_np.float32).tobytes():
            h = ((h ^ byte) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        out[fam] = dict(len=len(s), index=idx, value=[float(s[i]) for i in idx],
                        sigma_min=float(ms.sigma_min), sigma_max=float(ms.sigma_max),
                        bits_fnv1a=h)
    return out


def betaincinv_fixture():
    """`scipy.stats.beta.ppf` — the inverse regularized incomplete beta, which the `beta`
    scheduler needs and which has no closed form. a = b = 0.6 is ComfyUI's default; the
    other pairs (asymmetric, and a/b on either side of 1) exercise the branch structure
    of the continued fraction rather than just the symmetric case."""
    from scipy.stats import beta as _beta
    cases = []
    for a, b in ((0.6, 0.6), (0.6, 2.0), (2.0, 0.6), (1.0, 1.0), (5.0, 3.0), (0.2, 0.9)):
        for p in (1e-6, 0.01, 0.1, 0.25, 1.0 / 3.0, 0.5, 0.7, 0.9, 0.99, 1.0 - 1e-9, 1.0):
            cases.append(dict(a=a, b=b, p=p, x=float(_beta.ppf(p, a, b))))
    return cases


def main():
    const_ms = const_sampling()
    eps_ms = eps_sampling()

    # ⚠️ ComfyUI's OWN `normal` scheduler, not a reimplementation of it. This is the
    # fixture that catches the interpolation space: `ModelSamplingDiscrete.sigma`
    # lerps `log_sigmas` and exponentiates, where diffusers lerps sigma directly. The
    # two agree to 4.4e-5, which a diffusers-generated fixture compared at 2e-4
    # cannot distinguish — and which is 44x the Brownian tree's 1e-6 time quantum, so
    # an SDE sampler renders a completely different image on the wrong one.
    def sd_sigmas(steps):
        return [float(v) for v in comfy.samplers.normal_scheduler(eps_ms, steps)]

    schedules = {str(n): sd_sigmas(n) for n in (4, 8, 10, 20, 30)}

    trajectories = [
        # krea2 (CONST). sigmas[0] == 1.0 exactly, so this arm is the one that
        # exercises `offset_first_sigma_for_snr`.
        trajectory("krea2_heun_8", const_ms, simple_schedule(8), "heun", 1.0, 1.0),
        # eta = 0 makes it deterministic: no noise term at all, so this isolates the
        # 2M solver from the Brownian tree.
        trajectory("krea2_heun_8_eta0", const_ms, simple_schedule(8), "heun", 0.0, 1.0),
        trajectory("krea2_midpoint_8", const_ms, simple_schedule(8), "midpoint", 1.0, 1.0),
        # SD (EPS), where lambda = -log(sigma) and sigma_max is ~14.6.
        trajectory("sd_heun_10", eps_ms, sd_sigmas(10), "heun", 1.0, 1.0),
        trajectory("sd_heun_4", eps_ms, sd_sigmas(4), "heun", 1.0, 1.0),
        # A non-default eta/s_noise pair, so neither is accidentally hardcoded.
        trajectory("sd_heun_10_eta05", eps_ms, sd_sigmas(10), "heun", 0.5, 0.9),
    ]

    doc = dict(
        _comment="Generated by tools/gen_sampler_fixtures.py against ComfyUI's own "
                 "k_diffusion.sampling and torchsde. Do not hand-edit.",
        brownian=brownian_fixture(),
        sd_schedules=schedules,
        sigma_tables=table_fixture(const_ms, eps_ms),
        schedulers=scheduler_fixture(const_ms, eps_ms),
        betaincinv=betaincinv_fixture(),
        trajectories=trajectories,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")
    for t in trajectories:
        print(f"  {t['name']:22s} family={t['family']:5s} eta={t['eta']} "
              f"|x_out| max={max(abs(v) for v in t['x_out']):.4f}")
    for fam in ("flux", "sd"):
        for name in SCHEDULERS:
            got = doc["schedulers"][f"{fam}|{name}|20"]
            print(f"  {fam:4s} {name:17s} 20 steps -> {len(got):3d} sigmas  "
                  f"[{got[0]:.5f} .. {got[-2]:.6f}]")


if __name__ == "__main__":
    main()
