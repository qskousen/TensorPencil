"""Fixtures for `core/philox_rng.zig` — NVIDIA's Philox4x32-10 + Box-Muller noise, the
generator AUTOMATIC1111 draws its initial latent from.

A1111's `randn_source` defaults to **"GPU"** (real CUDA `torch.randn`), and it ships
`modules/rng_philox.py` — a pure-numpy CPU imitation of it — behind the "NV" setting.
That imitation is the executable reference used here: it is 100 lines, needs only numpy,
and reproduces CUDA's noise exactly for any tensor whose element count stays under the
launch's thread cap (see `philox_rng.zig`'s module doc for the exact threshold).

⚠️ **A1111 is AGPL-3.0, so its source is NOT vendored into this repository.** The
generator downloads it at run time (a maintainer-only tool, not distributed) and records
the upstream sha256, exactly as `gen_a1111_prompt_fixtures.py` does.

Two fixture groups:

  1. `philox_raw` — the bare `philox4_32` block cipher on hand-picked counter/key pairs.
     Pinned separately from the noise so a mismatch localizes to the rounds rather than
     to the Box-Muller transform.
  2. `philox_randn` — `Generator(seed).randn(shape)`, the whole pipeline. Full arrays for
     small draws; first/last 8 plus an FNV-1a hash over the raw f32 bits for latent-sized
     ones, which is how `schedule.zig`'s sigma tables are pinned.

⚠️ The arithmetic is **numpy's, not CUDA's**: `uint32 * float32` promotes to **float64**
under NumPy's promotion rules, so `box_muller` runs in f64 with f32-rounded constants and
narrows once at the end. curand's device code is f32 throughout and uses the `__sincosf`
intrinsic. The two agree to ~1e-6 relative, which is far below the 1.3e-4 model-level
disagreement CLAUDE.md already measures the render against — but it means this fixture
pins A1111's **NV** path bit-exactly and its **GPU** path only closely.

Usage (any env with numpy):
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_philox_fixtures.py
"""

import hashlib
import importlib.util
import json
import os
import tempfile
import urllib.request

import numpy as np

UPSTREAM = "https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/modules/rng_philox.py"
OUT = os.path.join(os.path.dirname(__file__), "..", "src", "core", "assets", "philox_fixtures.json")


def load_reference():
    """Download A1111's rng_philox and import it. Returns (module, sha256)."""
    src = urllib.request.urlopen(UPSTREAM, timeout=60).read()
    sha = hashlib.sha256(src).hexdigest()
    d = tempfile.mkdtemp(prefix="a1111rng-")
    path = os.path.join(d, "a1111_rng_philox.py")
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("a1111_rng_philox", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha


def fnv1a(a: np.ndarray) -> int:
    """FNV-1a over the raw f32 bits — the same digest `schedule.zig` pins tables with."""
    h = 0xCBF29CE484222325
    for b in np.ascontiguousarray(a, dtype=np.float32).tobytes():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def raw_cases(rp):
    """The block cipher alone, on counters that exercise every word."""
    cases = []
    for ctr, key in [
        ([0, 0, 0, 0], [0, 0]),
        ([0, 0, 1, 0], [0, 0]),
        ([1, 0, 0, 0], [0, 0]),
        ([0, 0, 12345, 0], [42, 0]),
        # A seed above 2^32, which is the only thing that exercises key[1].
        ([3, 0, 7, 0], [0xDEADBEEF, 0x12345678]),
        ([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF], [0xFFFFFFFF, 0xFFFFFFFF]),
    ]:
        c = np.array([[x] for x in ctr], dtype=np.uint32)
        k = np.array([[x] for x in key], dtype=np.uint32)
        out = rp.philox4_32(c, k)
        cases.append({
            "counter": [int(x) for x in ctr],
            "key": [int(x) for x in key],
            "out": [int(out[i][0]) for i in range(4)],
        })
    return cases


def randn_cases(rp):
    cases = []
    shapes = [
        (3, 4),          # the docstring's own example
        (16,),
        (255,),
        (256,),
        (257,),
        (4, 16, 16),     # 1024 — a 128x128 latent
        (4, 64, 64),     # 16384 — a 512x512 latent
        (4, 192, 128),   # 98304 — the 1024x1536 latent this was built for
    ]
    seeds = [0, 1, 42, 4294967295, 4294967296, 1234567890123456789]
    for shape in shapes:
        for seed in seeds:
            # Only the small shapes get the full seed sweep; the big ones are there to
            # pin the element mapping, which does not depend on the seed.
            if int(np.prod(shape)) > 1024 and seed not in (0, 42):
                continue
            g = rp.Generator(seed)
            a = g.randn(shape).ravel()
            case = {
                "seed": seed,
                "shape": list(shape),
                "n": int(a.size),
                "first8": [float(x) for x in a[:8]],
                "last8": [float(x) for x in a[-8:]],
                "hash": str(fnv1a(a)),
            }
            if a.size <= 64:
                case["all"] = [float(x) for x in a]
            # A second draw from the SAME generator, which advances `offset` — the
            # thing that distinguishes a stateful generator from a pure function.
            b = g.randn(shape).ravel()
            case["next_first8"] = [float(x) for x in b[:8]]
            case["next_hash"] = str(fnv1a(b))
            cases.append(case)
    return cases


def main():
    rp, sha = load_reference()

    # The docstring carries its own expected output; check the reference behaves as
    # documented before deriving anything from it.
    got = rp.Generator(seed=0).randn(shape=(3, 4))
    want = np.array([
        [-0.92466259, -0.42534415, -2.6438457, 0.14518388],
        [-0.12086647, -0.57972564, -0.62285122, -0.32838709],
        [-1.07454231, -0.36314407, -1.67105067, 2.26550497],
    ])
    assert np.allclose(got, want, atol=1e-7), f"reference does not match its own docstring:\n{got}"

    # And that the promotion really is f64 — if numpy ever changes this, the Zig port's
    # arithmetic would have to change with it, so fail loudly rather than silently
    # regenerate a fixture the port cannot match.
    probe = np.arange(4, dtype=np.uint32) * rp.two_pow32_inv
    assert probe.dtype == np.float64, f"box_muller no longer promotes to f64 (got {probe.dtype})"

    fx = {
        "_meta": {
            "source": UPSTREAM,
            "sha256": sha,
            "numpy": np.__version__,
            "note": "A1111 rng_philox = its randn_source='NV'; equals CUDA randn below the launch thread cap",
        },
        "philox_raw": raw_cases(rp),
        "philox_randn": randn_cases(rp),
    }
    with open(OUT, "w") as f:
        json.dump(fx, f, indent=1)
    print(f"wrote {OUT}: {len(fx['philox_raw'])} raw + {len(fx['philox_randn'])} randn cases")
    print(f"upstream sha256 {sha}")


if __name__ == "__main__":
    main()
