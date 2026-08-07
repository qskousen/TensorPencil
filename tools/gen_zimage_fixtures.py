#!/usr/bin/env python3
"""Reference fixtures for Z-Image (Tongyi's 6B `NextDiT`, the architecture "zit"
checkpoints use).

⚠️ **It is NOT Cosmos and NOT its own file in ComfyUI.** Z-Image runs through
`comfy/ldm/lumina/model.py` — the Lumina-Image-2.0 `NextDiT` — with
`z_image_modulation=True`, `pad_tokens_multiple=32`, `time_scale=1000` and
`rope_theta=256`. `comfy/supported_models.py::ZImage` subclasses `Lumina2` and is
selected purely by `dim == 3840`.

Reference is **ComfyUI**, for the reasons `gen_sdxl_fixtures.py` gives: it is the
compatibility target, and every convention that matters here is a *choice* the
checkpoint does not record — which hidden state of the text encoder conditions the
model, where the learned pad tokens go, what position ids the two halves of the
sequence get, and the sign of the output.

⚠️ **The config is DERIVED from ComfyUI, never hand-copied.** `model_detection`
reads it off the tensor shapes; this script asserts the values it then writes into
the fixture, so a future ComfyUI change that moves one of them fails here loudly
instead of silently re-baselining the Zig side against a stale constant.

## Why the DiT reference is truncated to 8 layers

A fp32 copy of the real 6.1B denoiser is 24.6 GB and does not fit in this machine's
RAM. Casting the reference to bf16 instead would put a ~1e-2 error floor in the
*reference*, which is coarser than several of the bugs this fixture exists to catch
(a wrong pad-token position and a wrong rope axis both survive 1e-2).

⚠️ **8 is what fits, and it was found by measurement, not guessed**: 16 layers is
~14.8 GB of fp32 weights before torch's own overhead and was OOM-killed inside a 22 GB
cgroup bound. Run this under `systemd-run --user --scope -p MemoryMax=22G
-p MemorySwapMax=0` so an over-estimate kills only this process — unbounded, it froze
the machine.

So the DiT is built directly at the detected config with `n_layers=REF_LAYERS` and loaded
from the real checkpoint's own tensors. That is an **exact fp32 reference on real
weights at real width** — it validates the loader, the naming, the two refiner
stacks, the modulation form, the rope layout, the padding and the output sign. What
it deliberately does *not* cover is "does the block loop run 30 times", which is a
loop bound, and which the end-to-end render comparison covers instead.

Usage (ComfyUI's `nvenv`, NOT the ai-toolkit venv):
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_zimage_fixtures.py
"""

import argparse
import hashlib
import json
import os
import sys

COMFY = "/home/qt/genai/comfyui"
DEFAULT_CKPT = os.path.join(COMFY, "models/checkpoints/zit/unstableRevolution_V2Fp16.safetensors")
DEFAULT_TE = os.path.join(COMFY, "models/text_encoders/qwen_3_4b.safetensors")
DEFAULT_VAE = os.path.join(COMFY, "models/vae/z-image-turbo.vae.safetensors")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_OUT = os.path.join(REPO, "src", "models", "assets", "zimage_ref.safetensors")

SEED = 20260806
PROMPTS = [
    "a photograph of an astronaut riding a horse",
    "",  # the unconditional branch; an empty prompt is still 9 template tokens
]
# 12x12 latent -> 6x6 = 36 image tokens, which is NOT a multiple of
# `pad_tokens_multiple` (32) — so this case exercises `x_pad_token` padding to 64.
# A size that happened to be a multiple would make a missing pad path pass.
LAT_SMALL = 12
# 32x32 -> 16x16 = 256 tokens, exactly 8x32, so pad_extra == 0. The pair pins that
# padding is applied when needed and *not* applied when it is not.
LAT_BIG = 32
# How many transformer blocks the fp32 reference keeps. See the module docstring.
REF_LAYERS = 8

# ⚠️ ComfyUI parses argv at import time; our own flags must come off `sys.argv`
# before `import comfy.*`. `--cpu` because a reference must not depend on a GPU
# reduction order, `--disable-xformers` because ComfyUI binds its attention
# implementation at import time and prefers xformers, which has no CPU fp32 kernel.
OUR_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu", "--fp32-vae", "--disable-smart-memory",
            "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

# ⚠️ Imported as a library, ComfyUI ignores argv entirely unless this is called
# first — `comfy/cli_args.py` parses `[]`. Without it every flag above silently
# defaults and the "CPU fp32" reference quietly runs on the GPU. The assert below
# is the only diagnostic there would otherwise be.
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
from comfy.cli_args import args as comfy_args  # noqa: E402
import comfy.model_management as _mm  # noqa: E402

if not (comfy_args.cpu and comfy_args.disable_xformers):
    raise SystemExit("ComfyUI did not take our flags — check comfy.options.enable_args_parsing()")
if _mm.xformers_enabled():
    raise SystemExit("xformers is still active; it has no CPU fp32 attention kernel")

import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.model_detection  # noqa: E402
from comfy.ldm.lumina.model import NextDiT  # noqa: E402


def stats(t: torch.Tensor) -> list:
    """(mean, l2, max_abs) — the localizing triple for a tensor too big to store."""
    f = t.detach().float().reshape(-1)
    return [float(f.mean()), float(f.norm()), float(f.abs().max())]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --- the config, read off the checkpoint by ComfyUI itself -------------------

# What this generator (and the Zig implementation) believes Z-Image is. Asserted
# against `model_detection`'s answer rather than trusted: if ComfyUI ever changes
# one of these, the fixture must not be regenerated silently against a new model.
EXPECTED = {
    "image_model": "lumina2",
    "patch_size": 2,
    "in_channels": 16,
    "dim": 3840,
    "cap_feat_dim": 2560,
    "n_layers": 30,
    "qk_norm": True,
    "n_heads": 30,
    "n_kv_heads": 30,
    "axes_dims": [32, 48, 48],
    "axes_lens": [1536, 512, 512],
    "rope_theta": 256.0,
    "z_image_modulation": True,
    "time_scale": 1000.0,
    "pad_tokens_multiple": 32,
}


class LazySafetensors:
    """A read-through mapping over a safetensors file.

    ⚠️ Loading the 12 GB checkpoint eagerly and then keeping a fp32 copy of the
    slice we want peaks well past this machine's free RAM. `detect_unet_config`
    only *indexes* two tensors (it works off the key list otherwise) and the
    truncated reference needs about a sixth of the file, so materializing on
    demand keeps the peak at roughly the size of what is actually used.
    """

    def __init__(self, path):
        from safetensors import safe_open

        self._f = safe_open(path, framework="pt", device="cpu")
        self._keys = list(self._f.keys())

    def keys(self):
        return self._keys

    def __contains__(self, k):
        return k in self._f.keys()

    def __getitem__(self, k):
        return self._f.get_tensor(k)

    def get(self, k, default=None):
        return self._f.get_tensor(k) if k in self._f.keys() else default

    def items(self):
        for k in self._keys:
            yield k, self._f.get_tensor(k)


def detect_config(sd) -> dict:
    cfg = comfy.model_detection.detect_unet_config(sd, "model.diffusion_model.", None)
    for k, want in EXPECTED.items():
        got = cfg.get(k)
        if isinstance(want, float):
            ok = got is not None and abs(float(got) - want) < 1e-9
        else:
            ok = got == want
        if not ok:
            raise SystemExit(
                f"ComfyUI detected {k}={got!r}, this generator expects {want!r}. "
                "The architecture moved — do not regenerate until the Zig side is updated.")
    # ffn_dim_multiplier is 8/3 and cannot be compared exactly against a literal.
    if abs(cfg["ffn_dim_multiplier"] - 8.0 / 3.0) > 1e-12:
        raise SystemExit(f"ffn_dim_multiplier={cfg['ffn_dim_multiplier']}, expected 8/3")
    return cfg


def build_ref_dit(sd, cfg: dict, n_layers: int) -> NextDiT:
    """A fp32 `NextDiT` at the real config but only `n_layers` blocks, loaded from
    the real checkpoint. See the module docstring for why it is truncated."""
    kwargs = {k: v for k, v in cfg.items() if k not in ("image_model", "allow_fp16")}
    kwargs["n_layers"] = n_layers
    model = NextDiT(**kwargs, device="cpu", dtype=torch.float32, operations=comfy.ops.disable_weight_init)

    pfx = "model.diffusion_model."
    want = set(model.state_dict().keys())
    have = {k[len(pfx):] for k in sd.keys() if k.startswith(pfx)}
    missing = want - have
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} tensors, e.g. {sorted(missing)[:5]}")
    model.load_state_dict({k: sd[pfx + k].float() for k in sorted(want)})
    return model.eval()


def run_text_encoder(te_path: str, out_path: str) -> None:
    """The Qwen3-4B stage — run in a CHILD PROCESS, and that is not tidiness.

    ⚠️ fp32 Qwen3-4B is 16 GB and ComfyUI's loader holds the bf16 source alive
    while it casts, so the peak is ~24 GB against this machine's ~23 GB free. Doing
    it in a child means the peak is never concurrent with the denoiser's, and the
    16 GB is returned to the OS the moment the stage ends rather than sitting in a
    fragmented heap for the rest of the run.
    """
    clip = comfy.sd.load_clip(
        ckpt_paths=[te_path],
        embedding_directory=None,
        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
        model_options={"dtype": torch.float32},
    )
    te_name = type(clip.cond_stage_model).__name__
    if "ZImage" not in te_name:
        raise SystemExit(f"ComfyUI picked {te_name} for this text encoder, not the Z-Image one")

    tensors: dict[str, torch.Tensor] = {}
    rows: dict[str, list] = {}
    with torch.no_grad():
        for i, text in enumerate(PROMPTS):
            tokens = clip.tokenize(text)
            ids = [int(t) for t, _w in tokens["qwen3_4b"][0]]
            cond = clip.encode_from_tokens(tokens, return_pooled=False).float()
            tensors[f"te.tokens.{i}"] = torch.tensor(ids, dtype=torch.int32)
            # f32 here: this tensor is also the DiT reference's *input*, so the
            # parent replays the DiT on exactly these values. Rounding it would put
            # a difference in the input of a comparison, not just its output.
            tensors[f"te.cond.{i}"] = cond[0]
            rows[f"te.cond.{i}"] = stats(cond)
    comfy.utils.save_torch_file(tensors, out_path, metadata={
        "stats": json.dumps(rows),
        "te_template": clip.tokenizer.llama_template,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--text-encoder", default=DEFAULT_TE)
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--te-stage", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(OUR_ARGV)

    if args.te_stage:
        run_text_encoder(args.text_encoder, args.te_stage)
        return

    for p in (args.checkpoint, args.text_encoder, args.vae):
        if not os.path.exists(p):
            raise SystemExit(f"not found: {p}")

    out: dict[str, torch.Tensor] = {}
    meta: dict[str, str] = {}
    stat_rows: dict[str, list] = {}

    meta["checkpoint"] = os.path.basename(args.checkpoint)
    meta["checkpoint_sha256"] = sha256(args.checkpoint)
    meta["text_encoder"] = os.path.basename(args.text_encoder)
    meta["text_encoder_sha256"] = sha256(args.text_encoder)
    meta["vae"] = os.path.basename(args.vae)
    meta["vae_sha256"] = sha256(args.vae)
    meta["ref_layers"] = str(REF_LAYERS)

    torch.manual_seed(SEED)

    # --- text encoder: Qwen3-4B, penultimate hidden state, no final norm -----
    import gc
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        te_tmp = os.path.join(td, "te.safetensors")
        print("running the text encoder stage (child process, fp32, ~16 GB)...")
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--text-encoder", args.text_encoder, "--te-stage", te_tmp],
                       check=True, cwd=REPO)
        te_sd, te_meta = comfy.utils.load_torch_file(te_tmp, return_metadata=True)

    conds = []
    for i in range(len(PROMPTS)):
        out[f"te.tokens.{i}"] = te_sd[f"te.tokens.{i}"]
        # f16 in the fixture: the Zig TE is compared at ~5e-3 relative, and two f32
        # [seq, 2560] tensors would be a third of the file for no extra strictness.
        half = te_sd[f"te.cond.{i}"].half()
        out[f"te.cond.{i}"] = half
        # ⚠️ The DiT reference is replayed on the ROUNDED conditioning, not the f32
        # one. The Zig DiT test reads `te.cond.i` out of this very fixture, so
        # feeding torch the unrounded values would put a 1e-3 difference in the
        # *input* of a comparison whose tolerance is 1e-3 — the exact trap the SDXL
        # generator's `clip.context` note calls out.
        conds.append(half.float().unsqueeze(0))
    stat_rows.update(json.loads(te_meta["stats"]))
    meta["te_template"] = te_meta["te_template"]
    del te_sd
    gc.collect()

    # --- the DiT -------------------------------------------------------------
    print("opening the denoiser state dict (lazily)...")
    sd = LazySafetensors(args.checkpoint)
    cfg = detect_config(sd)
    meta["config"] = json.dumps({k: cfg[k] for k in sorted(cfg) if k != "image_model"})
    print(f"building the fp32 reference at {REF_LAYERS} of {cfg['n_layers']} layers...")
    dit = build_ref_dit(sd, cfg, REF_LAYERS)
    del sd
    gc.collect()

    cap = {}

    def grab(name, idx=0):
        def fn(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            cap.setdefault(name, []).append(t.detach().float().clone())
        return fn

    handles = [
        dit.t_embedder.register_forward_hook(grab("dit.t_emb")),
        dit.cap_embedder.register_forward_hook(grab("dit.cap_embed")),
        dit.x_embedder.register_forward_hook(grab("dit.x_embed")),
        dit.rope_embedder.register_forward_hook(grab("dit.rope")),
        dit.final_layer.register_forward_hook(grab("dit.final")),
    ]
    for i, layer in enumerate(dit.context_refiner):
        handles.append(layer.register_forward_hook(grab(f"dit.ctx_refiner.{i}")))
    for i, layer in enumerate(dit.noise_refiner):
        handles.append(layer.register_forward_hook(grab(f"dit.noise_refiner.{i}")))
    for i, layer in enumerate(dit.layers):
        handles.append(layer.register_forward_hook(grab(f"dit.layer.{i}")))

    # Two (latent size, sigma) cases; case 0 pads the image half, case 1 does not.
    cases = [(LAT_SMALL, 0.75, 0), (LAT_BIG, 0.25, 1)]
    with torch.no_grad():
        for ci, (lat, sigma, prompt_i) in enumerate(cases):
            cap.clear()
            x = torch.randn(1, cfg["in_channels"], lat, lat, generator=torch.Generator().manual_seed(SEED + ci))
            ctx = conds[prompt_i].float()
            # ComfyUI passes `timesteps = model_sampling.timestep(sigma) = sigma`
            # (multiplier 1.0), and NextDiT._forward computes t = 1 - timesteps.
            ts = torch.tensor([sigma], dtype=torch.float32)
            v = dit(x, timesteps=ts, context=ctx, num_tokens=ctx.shape[1])

            out[f"dit.{ci}.x"] = x
            out[f"dit.{ci}.sigma"] = torch.tensor([sigma], dtype=torch.float32)
            out[f"dit.{ci}.cond_index"] = torch.tensor([prompt_i], dtype=torch.int32)
            out[f"dit.{ci}.out"] = v.float()
            for name, vals in cap.items():
                for j, t in enumerate(vals):
                    key = f"dit.{ci}.{name}" + (f".{j}" if len(vals) > 1 else "")
                    stat_rows[key] = stats(t)
                    # Full tensors only for the small case, and only f16: the big
                    # case is [1, 288, 3840] per stage and would dominate the file.
                    if ci == 0:
                        out[key] = t.squeeze(0).half()
    for h in handles:
        h.remove()
    del dit
    gc.collect()

    # --- the VAE decoder (AutoencoderKL, 16 latent channels) -----------------
    print("loading the VAE...")
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))
    vae.first_stage_model = vae.first_stage_model.float().eval()
    with torch.no_grad():
        for ci, zl in enumerate((8, 32)):
            z = torch.randn(1, 16, zl, zl, generator=torch.Generator().manual_seed(SEED + 100 + ci))
            # `vae.decode` applies `process_out` (the Flux scale/shift) itself; the
            # Zig decoder takes the already-unscaled latent, so feed the raw tensor
            # through the inner model to keep the comparison to the decoder alone.
            img = vae.first_stage_model.decode(z.float())
            out[f"vae.{ci}.z"] = z
            out[f"vae.{ci}.rgb"] = img.float()
            stat_rows[f"vae.{ci}.rgb"] = stats(img)
    meta["vae_scale_factor"] = repr(vae.latent_format.scale_factor if hasattr(vae, "latent_format") else 0.3611)

    meta["stats"] = json.dumps(stat_rows)
    os.makedirs(os.path.dirname(REF_OUT), exist_ok=True)
    comfy.utils.save_torch_file(out, REF_OUT, metadata=meta)
    total = sum(v.numel() * v.element_size() for v in out.values())
    print(f"wrote {REF_OUT} ({total / 1e6:.1f} MB, {len(out)} tensors, {len(stat_rows)} stat rows)")


if __name__ == "__main__":
    main()
