#!/usr/bin/env python3
"""Reference fixtures for the stochastic samplers — DPM++ 2M SDE (Heun and midpoint)
and Euler ancestral — each with its own noise source.

Needs a CUDA device: the ancestral samplers draw on the latent's own device, so a
CPU-only run would pin the wrong generator (see the `2b` section).

Tiers, because very different things need pinning:

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
        # The samplers reach for the model_sampling object through two DIFFERENT
        # attribute chains, so the stub has to present both: `sample_dpmpp_2m_sde`
        # goes through `inner_model.model_patcher.get_model_object`, while
        # `sample_euler_ancestral`'s CONST dispatch reads
        # `inner_model.inner_model.model_sampling` directly.
        class _Patcher:
            def get_model_object(self_inner, name):
                assert name == "model_sampling"
                return model_sampling
        class _Innermost:
            pass
        _Innermost.model_sampling = model_sampling
        class _Inner:
            model_patcher = _Patcher()
            inner_model = _Innermost()
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
    fn = {
        "heun": kds.sample_dpmpp_2m_sde_heun,
        "midpoint": kds.sample_dpmpp_2m_sde,
        "3m": kds.sample_dpmpp_3m_sde,
    }[solver_type]
    out = fn(model, x0.clone(), sig, extra_args={"seed": SEED},
             disable=True, eta=eta, s_noise=s_noise)
    return dict(name=name, family=("const" if isinstance(model_sampling, ms.CONST) else "eps"),
                solver_type=solver_type, eta=eta, s_noise=s_noise, seed=SEED, n=N,
                sigmas=[float(v) for v in sig], c=f32_list(c), x0=f32_list(x0),
                x_out=f32_list(out))


def plain_trajectory(name, fn, model_sampling, sigmas, **kw):
    """A sampler that takes no eta and draws nothing: same shape as `trajectory`, on
    the CPU, since there is no generator whose device could matter."""
    torch.manual_seed(0)
    c = torch.randn(1, 4, 4, 4, dtype=torch.float32)
    x0 = torch.randn(1, 4, 4, 4, dtype=torch.float32)
    sig = torch.tensor(sigmas, dtype=torch.float32)
    out = fn(ToyDenoiser(c, model_sampling), x0.clone(), sig,
             extra_args={"seed": SEED}, disable=True, **kw)
    return dict(name=name, family=("const" if isinstance(model_sampling, ms.CONST) else "eps"),
                eta=0.0, s_noise=1.0, seed=SEED, n=N,
                sigmas=[float(v) for v in sig], c=f32_list(c), x0=f32_list(x0),
                x_out=f32_list(out))


# --- 2b. The ancestral samplers, on the GPU ---------------------------------------
#
# ⚠️ These must run on CUDA. `default_noise_sampler` builds its generator on
# `x.device`, so a CPU run would pin torch's MT19937 *and* the `seed + 1` the CPU
# branch applies, neither of which is what a ComfyUI user renders with. The Brownian
# tree above is the opposite case: ComfyUI forces it to the CPU with `cpu=True`.


def ancestral_trajectory(name, model_sampling, sigmas, eta, s_noise):
    torch.manual_seed(0)
    c = torch.randn(1, 4, 4, 4, dtype=torch.float32).cuda()
    x0 = torch.randn(1, 4, 4, 4, dtype=torch.float32).cuda()
    sig = torch.tensor(sigmas, dtype=torch.float32).cuda()

    model = ToyDenoiser(c, model_sampling)
    # Called through the dispatching entry point, not the `_RF` one: which arm a
    # family takes is part of what is under test.
    out = kds.sample_euler_ancestral(model, x0.clone(), sig, extra_args={"seed": SEED},
                                     disable=True, eta=eta, s_noise=s_noise)
    return dict(name=name, family=("const" if isinstance(model_sampling, ms.CONST) else "eps"),
                eta=eta, s_noise=s_noise, seed=SEED, n=N,
                sigmas=[float(v) for v in sig.cpu()], c=f32_list(c.cpu()), x0=f32_list(x0.cpu()),
                x_out=f32_list(out.cpu()))


def trajectory_2s(name, model_sampling, sigmas, eta, s_noise):
    """`dpmpp_sde`: two evaluations and a Brownian tree, which ComfyUI forces to the
    CPU, so this one stays off the GPU like the other tree samplers."""
    torch.manual_seed(0)
    c = torch.randn(1, 4, 4, 4, dtype=torch.float32)
    x0 = torch.randn(1, 4, 4, 4, dtype=torch.float32)
    sig = torch.tensor(sigmas, dtype=torch.float32)
    out = kds.sample_dpmpp_sde(ToyDenoiser(c, model_sampling), x0.clone(), sig,
                               extra_args={"seed": SEED}, disable=True, eta=eta, s_noise=s_noise)
    return dict(name=name, family=("const" if isinstance(model_sampling, ms.CONST) else "eps"),
                eta=eta, s_noise=s_noise, seed=SEED, n=N,
                sigmas=[float(v) for v in sig], c=f32_list(c), x0=f32_list(x0),
                x_out=f32_list(out))


def ancestral2_trajectory(name, fn, model_sampling, sigmas, eta, s_noise):
    """The two-evaluation ancestral samplers, on CUDA for the same reason as the
    one-evaluation one: `default_noise_sampler` follows the latent's device."""
    torch.manual_seed(0)
    c = torch.randn(1, 4, 4, 4, dtype=torch.float32).cuda()
    x0 = torch.randn(1, 4, 4, 4, dtype=torch.float32).cuda()
    sig = torch.tensor(sigmas, dtype=torch.float32).cuda()
    out = fn(ToyDenoiser(c, model_sampling), x0.clone(), sig, extra_args={"seed": SEED},
             disable=True, eta=eta, s_noise=s_noise)
    return dict(name=name, family=("const" if isinstance(model_sampling, ms.CONST) else "eps"),
                eta=eta, s_noise=s_noise, seed=SEED, n=N,
                sigmas=[float(v) for v in sig.cpu()], c=f32_list(c.cpu()), x0=f32_list(x0.cpu()),
                x_out=f32_list(out.cpu()))


def cuda_randn_seq(draws=4):
    """Successive `torch.randn` from ONE `torch.Generator(device="cuda")`, which is
    exactly what `default_noise_sampler` holds for a whole render.

    Pins the *sequence*, not just the values: `philox_rng.zig`'s counter advances one
    block per draw, and a generator that restarted per draw would give every step the
    same field while still looking like noise."""
    g = torch.Generator(device="cuda")
    g.manual_seed(SEED)
    return dict(seed=SEED, n=N,
                draws=[f32_list(torch.randn(N, device="cuda", generator=g, dtype=torch.float32).cpu())
                       for _ in range(draws)])


def const_sampling(shift=1.15):
    """krea2 / flux: `ModelSamplingFlux` + `CONST`, exactly what `model_sampling()`
    composes for `ModelType.FLUX`."""
    class MS(ms.ModelSamplingFlux, ms.CONST):
        pass
    m = MS()
    m.set_parameters(shift=shift)
    return m


def zimage_sampling(shift=3.0):
    """Z-Image / Lumina 2: `ModelSamplingDiscreteFlow` + `CONST`, with the
    `sampling_settings` `supported_models.py::ZImage` declares (shift 3.0,
    multiplier 1.0) — which the official ComfyUI template also sets explicitly via a
    `ModelSamplingAuraFlow` node.

    ⚠️ Its table is **1000** rungs where `ModelSamplingFlux`'s is 10000, and its sigma
    formula is the same function written as a single tensor division rather than
    `scalar / tensor`. Both differences are invisible to Euler and decisive for an SDE
    sampler, whose Brownian tree keys on the sigma quantised to 1e-6.
    """
    class MS(ms.ModelSamplingDiscreteFlow, ms.CONST):
        pass
    m = MS()
    m.set_parameters(shift=shift, multiplier=1.0)
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


def scheduler_fixture(const_ms, eps_ms, zimg_ms):
    """ComfyUI's `calculate_sigmas` for every scheduler x both families x several step
    counts, called through ComfyUI's own dispatcher (which owns the `use_ms` split
    between `f(model_sampling, steps)` and `f(n, sigma_min, sigma_max)`).

    ⚠️ Several of these do NOT return `steps + 1` sigmas: `ddim_uniform` strides the
    table, and `beta` de-duplicates repeated indices. The count is emitted so the Zig
    side asserts it rather than assuming, and so the sampling loop is reminded to take
    its step count from the schedule.
    """
    out = {}
    for fam, ms in (("flux", const_ms), ("sd", eps_ms), ("zimage", zimg_ms)):
        for name in SCHEDULERS:
            for steps in SCHED_STEPS:
                sig = comfy.samplers.calculate_sigmas(ms, name, steps)
                out[f"{fam}|{name}|{steps}"] = [float(v) for v in sig]
    return out


def table_fixture(const_ms, eps_ms, zimg_ms):
    """The two `model_sampling.sigmas` tables' own endpoints and a few interior entries.

    ⚠️ These are computed by torch in **f32** (`ModelSamplingFlux.set_parameters` builds
    them from an f32 `arange/timesteps` tensor), so a Zig port computing the same formula
    in f64 lands a fraction of an ulp away — which is invisible to Euler and decisive for
    the Brownian tree. Pinned explicitly so the interpolation-space lesson is not
    re-learned on the flux table.
    """
    out = {}
    for fam, ms in (("flux", const_ms), ("sd", eps_ms), ("zimage", zimg_ms)):
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
    zimg_ms = zimage_sampling()

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
        # DPM++(3M) SDE. 10 steps at least, since its first two steps degrade to
        # first and second order and only step 3 onward exercises the 3M correction.
        trajectory("krea2_3m_10", const_ms, simple_schedule(10), "3m", 1.0, 1.0),
        trajectory("sd_3m_10", eps_ms, sd_sigmas(10), "3m", 1.0, 1.0),
        trajectory("sd_3m_10_eta05", eps_ms, sd_sigmas(10), "3m", 0.5, 0.9),
        # eta = 0 leaves the 3M solver with no noise at all, which isolates the
        # third-order correction from the Brownian tree.
        trajectory("sd_3m_10_eta0", eps_ms, sd_sigmas(10), "3m", 0.0, 1.0),
    ]

    # heun / dpm_2: deterministic, two evaluations a step, one body for both families.
    plain_extra = [
        plain_trajectory("krea2_heun2_8", kds.sample_heun, const_ms, simple_schedule(8)),
        plain_trajectory("sd_heun2_10", kds.sample_heun, eps_ms, sd_sigmas(10)),
        plain_trajectory("krea2_dpm2_8", kds.sample_dpm_2, const_ms, simple_schedule(8)),
        plain_trajectory("sd_dpm2_10", kds.sample_dpm_2, eps_ms, sd_sigmas(10)),
    ]

    # The two-evaluation ancestral samplers, both arms each. `dpmpp_2s_ancestral` on
    # krea2 is the one that exercises the hardcoded 0.9999 probe the RF body falls back
    # to when the schedule starts at exactly sigma 1.
    ancestral2 = [
        ancestral2_trajectory("krea2_dpm2a_8", kds.sample_dpm_2_ancestral, const_ms, simple_schedule(8), 1.0, 1.0),
        ancestral2_trajectory("sd_dpm2a_10", kds.sample_dpm_2_ancestral, eps_ms, sd_sigmas(10), 1.0, 1.0),
        ancestral2_trajectory("sd_dpm2a_10_eta05", kds.sample_dpm_2_ancestral, eps_ms, sd_sigmas(10), 0.5, 0.9),
        ancestral2_trajectory("krea2_2sa_8", kds.sample_dpmpp_2s_ancestral, const_ms, simple_schedule(8), 1.0, 1.0),
        ancestral2_trajectory("sd_2sa_10", kds.sample_dpmpp_2s_ancestral, eps_ms, sd_sigmas(10), 1.0, 1.0),
        ancestral2_trajectory("sd_2sa_10_eta05", kds.sample_dpmpp_2s_ancestral, eps_ms, sd_sigmas(10), 0.5, 0.9),
        # eta = 0 still DRAWS in both of these (the reference calls the noise sampler
        # and scales it by zero), so it pins the draw count as much as the arithmetic.
        ancestral2_trajectory("sd_2sa_10_eta0", kds.sample_dpmpp_2s_ancestral, eps_ms, sd_sigmas(10), 0.0, 1.0),
    ]

    # dpmpp_sde: Brownian tree, so CPU like the other tree samplers.
    sde_2s = [
        trajectory_2s("krea2_sde_8", const_ms, simple_schedule(8), 1.0, 1.0),
        trajectory_2s("sd_sde_10", eps_ms, sd_sigmas(10), 1.0, 1.0),
        trajectory_2s("sd_sde_10_eta05", eps_ms, sd_sigmas(10), 0.5, 0.9),
    ]

    # DPM++(2M): deterministic, and the one sampler with no CONST dispatch at all,
    # so BOTH families run `t = -log(sigma)`. The krea2 entry is the one that would
    # catch a port that reached for the model's own half-logSNR here.
    plain = [
        plain_trajectory("krea2_2m_8", kds.sample_dpmpp_2m, const_ms, simple_schedule(8)),
        plain_trajectory("sd_2m_10", kds.sample_dpmpp_2m, eps_ms, sd_sigmas(10)),
        plain_trajectory("sd_2m_4", kds.sample_dpmpp_2m, eps_ms, sd_sigmas(4)),
    ] + plain_extra

    ancestral = [
        # krea2 (CONST), which takes the `_RF` arm: alpha moves with sigma, so the
        # step is a lerp toward `denoised` plus a rescale, not a variance split.
        ancestral_trajectory("krea2_ancestral_8", const_ms, simple_schedule(8), 1.0, 1.0),
        ancestral_trajectory("krea2_ancestral_8_eta05", const_ms, simple_schedule(8), 0.5, 0.9),
        # SD (EPS), the `get_ancestral_step` arm.
        ancestral_trajectory("sd_ancestral_10", eps_ms, sd_sigmas(10), 1.0, 1.0),
        ancestral_trajectory("sd_ancestral_4", eps_ms, sd_sigmas(4), 1.0, 1.0),
        ancestral_trajectory("sd_ancestral_10_eta05", eps_ms, sd_sigmas(10), 0.5, 0.9),
        # eta = 0 takes `get_ancestral_step`'s early out, which is what makes this
        # arm bit-identical to plain Euler rather than merely equal to it.
        ancestral_trajectory("sd_ancestral_10_eta0", eps_ms, sd_sigmas(10), 0.0, 1.0),
    ]

    doc = dict(
        _comment="Generated by tools/gen_sampler_fixtures.py against ComfyUI's own "
                 "k_diffusion.sampling and torchsde. Do not hand-edit.",
        brownian=brownian_fixture(),
        cuda_randn_seq=cuda_randn_seq(),
        sd_schedules=schedules,
        sigma_tables=table_fixture(const_ms, eps_ms, zimg_ms),
        schedulers=scheduler_fixture(const_ms, eps_ms, zimg_ms),
        betaincinv=betaincinv_fixture(),
        trajectories=trajectories,
        plain=plain,
        ancestral=ancestral,
        ancestral2=ancestral2,
        sde_2s=sde_2s,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")
    for t in trajectories + plain + ancestral + ancestral2 + sde_2s:
        print(f"  {t['name']:22s} family={t['family']:5s} eta={t['eta']} "
              f"|x_out| max={max(abs(v) for v in t['x_out']):.4f}")
    for fam in ("flux", "sd", "zimage"):
        for name in SCHEDULERS:
            got = doc["schedulers"][f"{fam}|{name}|20"]
            print(f"  {fam:4s} {name:17s} 20 steps -> {len(got):3d} sigmas  "
                  f"[{got[0]:.5f} .. {got[-2]:.6f}]")


if __name__ == "__main__":
    main()
