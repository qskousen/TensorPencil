#!/usr/bin/env python3
"""Reference fixtures for Anima — CircleStone/Comfy Org's 2B anime text-to-image
model.

⚠️ **Anima IS Cosmos-Predict2's DiT.** `comfy/ldm/anima/model.py` subclasses
`MiniTrainDIT` from `comfy/ldm/cosmos/predict2.py` and adds an `LLMAdapter`;
`model_detection` picks `"anima"` over `"cosmos_predict2"` purely by the presence
of `llm_adapter.blocks.0.cross_attn.q_proj.weight`.

Reference is **ComfyUI**, for the reason the SDXL and Z-Image generators give: it
is the compatibility target, and every convention that matters here is a *choice*
the checkpoint does not record — the patch feature orders (which differ between
input and output!), the all-zero padding-mask channel, whether the timestep is the
sigma or `1 - sigma` scaled by 1000, which sublayers share the AdaLN-LoRA vector,
and the RoPE axis order and NTK scaling.

⚠️ **The config is DERIVED from ComfyUI, never hand-copied.** `detect_unet_config`
reads it off the tensor shapes; this script asserts the values it then writes, so an
upstream change fails here loudly instead of silently re-baselining the Zig side
against a stale constant.

## What is truncated, what is not, and why

The whole model is ~2.1 B parameters, so a fp32 copy is ~8.4 GB — and ComfyUI's
loader holds the bf16 source alive while it casts. Two different decisions:

* **The `llm_adapter` is referenced IN FULL** (all 6 blocks). It is 133 M
  parameters, 0.5 GB in fp32, and it is a whole sub-model with its own tokenizer,
  its own embedding and two RoPE tables — truncating the cheap half of a model to
  save memory that is not scarce would be a pointless gap.
* **The DiT trunk is truncated to `REF_LAYERS` blocks**, and it is referenced at
  the real width on the real weights, which is what makes it an *exact fp32*
  reference. Casting to bf16 instead would put a ~1e-2 floor in the reference,
  coarser than several of the bugs this exists to catch (a swapped patch order and
  a wrong RoPE axis both survive 1e-2). A second depth is emitted so the trend is
  visible: a disagreement that grows LINEARLY with depth says the block is right,
  where one that compounds says it is merely close.

What truncation does not cover is the loop bound itself — the end-to-end render
comparison covers that.

⚠️ Run it under a memory bound, so an over-estimate kills only this process:
    systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_anima_fixtures.py
"""

import argparse
import hashlib
import json
import os
import sys

COMFY = "/home/qt/genai/comfyui"
DEFAULT_CKPT = os.path.join(
    COMFY, "models/diffusion_models/anima/terraRising_20TerraRisingAnima.safetensors")
DEFAULT_TE = os.path.join(COMFY, "models/text_encoders/qwen_3_06b_base.safetensors")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_OUT = os.path.join(REPO, "src", "models", "assets", "anima_ref.safetensors")

SEED = 20260807
# The real workflow's prompts, so the fixture covers what the render comparison
# runs — plus the empty one, which is the CFG unconditional branch and the case
# where the T5 branch is a single `</s>` and the adapter's 512-row pad is almost
# the entire context.
PROMPTS = [
    'score_8, masterpiece, newest, absurdres, incredibly absurdres, best quality, '
    'amazing quality, very aesthetic, solo, chromatic aberration, rainbow hair, '
    'black hair, (black theme), hair bun, limited palette, tenebrism, close up, '
    'depth of field, plump lips, victorian maid dress, elegant clothes, closed mouth, '
    'swept bangs, from side, looking to the side, dark red background, large breasts, '
    'cleavage, (lipstick writing on breasts says "Lilith\'s Desire")',
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    "",
]
# ⚠️ NON-SQUARE and not multiples of each other, deliberately: the DiT's RoPE has
# separate h and w axes whose thetas are EQUAL, so a square latent cannot tell an
# h/w transposition from the identity. 12x10 and 40x32 also give token counts
# (30 and 320) that are not powers of two.
LATENTS = [(12, 10), (40, 32)]
# Blocks the fp32 reference keeps, and a shallower depth for the trend.
REF_LAYERS = 8
TREND_LAYERS = 2

# ⚠️ ComfyUI parses argv at import time; our own flags come off `sys.argv` first.
# `--cpu` because a reference must not depend on a GPU reduction order,
# `--disable-xformers` because ComfyUI binds its attention implementation at import
# time and prefers xformers, which has no CPU fp32 kernel.
OUR_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory",
            "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

# ⚠️ Imported as a library, ComfyUI ignores argv entirely unless this is called
# first — `comfy/cli_args.py` parses `[]`. Without it every flag above silently
# defaults and the "CPU fp32" reference quietly runs on the GPU. The assert below is
# the only diagnostic there would otherwise be.
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
import comfy.supported_models  # noqa: E402
from comfy.ldm.anima.model import Anima  # noqa: E402

# What this generator (and `src/models/anima.zig`) believes Anima is. Asserted
# against `model_detection`'s answer rather than trusted.
EXPECTED = {
    "image_model": "anima",
    "in_channels": 16,
    "out_channels": 16,
    "model_channels": 2048,
    "num_blocks": 28,
    "num_heads": 16,
    "patch_spatial": 2,
    "patch_temporal": 1,
    "concat_padding_mask": True,
    "crossattn_emb_channels": 1024,
    "use_adaln_lora": True,
    "adaln_lora_dim": 256,
    "extra_per_block_abs_pos_emb": False,
    "rope_h_extrapolation_ratio": 4.0,
    "rope_w_extrapolation_ratio": 4.0,
    "rope_t_extrapolation_ratio": 1.0,
    "rope_enable_fps_modulation": False,
    "pos_emb_cls": "rope3d",
}
# `sampling_settings`, which is where conventions 3 (the timestep) and the sigma
# table's shift come from. Asserted against `supported_models.Anima` for the same
# reason as the unet config.
EXPECTED_SAMPLING = {"multiplier": 1.0, "shift": 3.0}


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


class LazySafetensors:
    """A read-through mapping over a safetensors file: `detect_unet_config` only
    *indexes* a couple of tensors and the truncated reference needs a fraction of
    the file, so materializing on demand keeps the peak near what is actually used."""

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


def detect_config(sd) -> dict:
    cfg = comfy.model_detection.detect_unet_config(sd, "model.diffusion_model.", None)
    for k, want in EXPECTED.items():
        got = cfg.get(k)
        ok = (abs(float(got) - want) < 1e-9) if isinstance(want, float) and got is not None else got == want
        if not ok:
            raise SystemExit(
                f"ComfyUI detected {k}={got!r}, this generator expects {want!r}. "
                "The architecture moved — do not regenerate until the Zig side is updated.")
    for k, want in EXPECTED_SAMPLING.items():
        got = comfy.supported_models.Anima.sampling_settings.get(k)
        if got != want:
            raise SystemExit(f"Anima.sampling_settings[{k}] is {got!r}, expected {want!r}")
    return cfg


def build_ref(sd, cfg: dict, n_layers: int) -> Anima:
    """A fp32 `Anima` at the real config with `n_layers` trunk blocks and the FULL
    6-block adapter, loaded from the real checkpoint."""
    kwargs = {k: v for k, v in cfg.items() if k not in ("image_model",)}
    kwargs["num_blocks"] = n_layers
    model = Anima(**kwargs, device="cpu", dtype=torch.float32,
                  operations=comfy.ops.disable_weight_init)

    pfx = "model.diffusion_model."
    want = set(model.state_dict().keys())
    have = {k[len(pfx):] for k in sd.keys() if k.startswith(pfx)}
    missing = want - have
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} tensors, e.g. {sorted(missing)[:5]}")
    model.load_state_dict({k: sd[pfx + k].float() for k in sorted(want)})
    return model.eval()


def run_text_encoder(te_path: str, out_path: str) -> None:
    """The Qwen3-0.6B stage, in a CHILD PROCESS so its fp32 copy is never concurrent
    with the denoiser's and is returned to the OS when the stage ends.

    ⚠️ This emits the ENCODER's output, not the adapter's. Keeping the two stages
    apart is the whole point: the encoder is a naming/tap/eps question (is it the
    final hidden state? with `model.norm` applied?) while the adapter is a
    cross-attention-wiring question. One combined tensor would say only "wrong"."""
    clip = comfy.sd.load_clip(
        ckpt_paths=[te_path],
        embedding_directory=None,
        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
        model_options={"dtype": torch.float32},
    )
    te_name = type(clip.cond_stage_model).__name__
    if "Anima" not in te_name:
        raise SystemExit(f"ComfyUI picked {te_name} for this text encoder, not the Anima one")

    tensors: dict[str, torch.Tensor] = {}
    rows: dict[str, list] = {}
    with torch.no_grad():
        for i, text in enumerate(PROMPTS):
            tokens = clip.tokenize(text)
            qwen_ids = [int(t) for t, _w in tokens["qwen3_06b"][0]]
            t5_ids = [int(t) for t, _w in tokens["t5xxl"][0]]
            t5_w = [float(w) for _t, w in tokens["t5xxl"][0]]

            # `encode_from_tokens` would run the adapter too (Anima.extra_conds calls
            # preprocess_text_embeds under inference mode). Go through the tower
            # directly so this tensor is the encoder's own output.
            cond, _pooled, extra = clip.cond_stage_model.encode_token_weights(tokens)
            # Cross-check that ComfyUI put the same T5 ids in `extra` that we read
            # off the tokenizer — the DiT stage consumes `extra`'s copy.
            got_ids = [int(v) for v in extra["t5xxl_ids"]]
            if got_ids != t5_ids:
                raise SystemExit("t5xxl_ids from encode_token_weights disagree with the tokenizer")

            # f32: this tensor is also the adapter reference's *input*, so the parent
            # replays on exactly these values. Rounding it would put a difference in
            # the input of a comparison rather than in its output.
            tensors[f"te.qwen_ids.{i}"] = torch.tensor(qwen_ids, dtype=torch.int32)
            tensors[f"te.t5_ids.{i}"] = torch.tensor(t5_ids, dtype=torch.int32)
            tensors[f"te.t5_weights.{i}"] = torch.tensor(t5_w, dtype=torch.float32)
            tensors[f"te.cond.{i}"] = cond[0].float()
            rows[f"te.cond.{i}"] = stats(cond)
    comfy.utils.save_torch_file(tensors, out_path, metadata={"stats": json.dumps(rows)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--text-encoder", default=DEFAULT_TE)
    ap.add_argument("--out", default=REF_OUT)
    ap.add_argument("--te-stage", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(OUR_ARGV)

    if args.te_stage is not None:
        run_text_encoder(args.text_encoder, args.te_stage)
        return

    torch.manual_seed(SEED)
    te_tmp = args.out + ".te-stage.safetensors"
    import subprocess
    subprocess.run([sys.executable, os.path.abspath(__file__),
                    "--text-encoder", args.text_encoder, "--te-stage", te_tmp],
                   check=True, cwd=REPO)
    te = comfy.utils.load_torch_file(te_tmp)
    os.remove(te_tmp)

    sd = LazySafetensors(args.checkpoint)
    cfg = detect_config(sd)

    tensors: dict[str, torch.Tensor] = dict(te)
    rows: dict[str, list] = {k: stats(v) for k, v in te.items() if v.dtype.is_floating_point}

    # ⚠️ The per-case INPUTS are drawn once, here, and reused at every depth. Drawing
    # them inside the depth loop made the shallow arm a different random latent, so the
    # depth-trend comparison — the whole reason a second depth exists — was between two
    # unrelated forwards. The Zig test reads these from the depth-8 tag for that reason.
    inputs = []
    for ci, (lat_h, lat_w) in enumerate(LATENTS):
        inputs.append({
            "lat_h": lat_h,
            "lat_w": lat_w,
            # prompt index: case 0 uses the positive prompt, case 1 the negative
            "pi": ci,
            "x": torch.randn(1, cfg["in_channels"], 1, lat_h, lat_w, dtype=torch.float32),
            # ⚠️ Convention 3: `model_sampling.timestep(sigma) = sigma * 1.0`, so the
            # argument is the sigma itself. Deliberately NOT 0.5 — a value where
            # `sigma` and `1 - sigma` coincide would hide the convention.
            "sigma": torch.tensor([0.8 if ci == 0 else 0.15], dtype=torch.float32),
        })

    for depth_i, n_layers in enumerate((REF_LAYERS, TREND_LAYERS)):
        model = build_ref(sd, cfg, n_layers)
        dit = model  # Anima IS the diffusion model
        d = cfg["model_channels"]

        with torch.no_grad():
            for ci, inp in enumerate(inputs):
                lat_h, lat_w, pi = inp["lat_h"], inp["lat_w"], inp["pi"]
                x, sigma = inp["x"], inp["sigma"]
                cond = te[f"te.cond.{pi}"].unsqueeze(0)
                t5_ids = te[f"te.t5_ids.{pi}"].unsqueeze(0).to(torch.int32)
                t5_w = te[f"te.t5_weights.{pi}"].unsqueeze(0).unsqueeze(-1)

                # --- the adapter, at FULL depth ---------------------------------
                adapter_raw = dit.llm_adapter(cond, t5_ids)
                ctx = dit.preprocess_text_embeds(cond, t5_ids, t5xxl_weights=t5_w)

                # --- the timestep path ------------------------------------------
                t_BT = sigma.unsqueeze(1)
                sinus = dit.t_embedder[0](t_BT).to(torch.float32)
                emb_raw, lora = dit.t_embedder[1](sinus)
                emb = dit.t_embedding_norm(emb_raw)
                # Block 0's three modulation vectors, each already summed with the
                # shared LoRA vector (convention 4).
                mod0 = torch.cat([
                    dit.blocks[0].adaln_modulation_self_attn(emb) + lora,
                    dit.blocks[0].adaln_modulation_cross_attn(emb) + lora,
                    dit.blocks[0].adaln_modulation_mlp(emb) + lora,
                ], dim=-1)
                # The final layer's, which takes only the FIRST 2*dim of the LoRA.
                mod_final = dit.final_layer.adaln_modulation(emb) + lora[:, :, : 2 * d]

                # --- x_embedder + the RoPE table --------------------------------
                # `prepare_embedded_sequence` is what appends the all-zero
                # padding-mask channel (convention 2) and patchifies, so calling it
                # is what pins that channel's existence and position. It also returns
                # the RoPE table, so there is no second construction to get wrong.
                x_emb, rope, extra_pos = dit.prepare_embedded_sequence(
                    x, fps=None, padding_mask=None)
                if extra_pos is not None:
                    raise SystemExit("extra_per_block_abs_pos_emb fired; the Zig side has no such path")
                # `[L, head_dim/2, 2, 2]` rotation matrices, stacked as
                # [[cos, -sin], [sin, cos]] — so cos is [...,0,0] and sin is [...,1,0].
                rope_cos = rope[..., 0, 0].contiguous()
                rope_sin = rope[..., 1, 0].contiguous()

                # --- block 0, and the whole forward -----------------------------
                blk0 = dit.blocks[0](
                    x_emb, emb, ctx,
                    rope_emb_L_1_1_D=rope.unsqueeze(1).unsqueeze(0),
                    adaln_lora_B_T_3D=lora,
                )
                out = dit(x, sigma, ctx)

                tag = f"dit{n_layers}.{ci}"
                if depth_i == 0:
                    tensors[f"{tag}.x"] = x[0, :, 0]
                    tensors[f"{tag}.sigma"] = sigma
                    tensors[f"{tag}.prompt_index"] = torch.tensor([pi], dtype=torch.int32)
                    tensors[f"{tag}.adapter_raw"] = adapter_raw[0].float()
                    # ⚠️ f32, NOT f16: this tensor is the DiT reference's own input, so the
                    # Zig side replays `predict` on exactly these values. Rounding it
                    # would put a difference in the INPUT of the comparison.
                    tensors[f"{tag}.ctx"] = ctx[0].float()
                    tensors[f"{tag}.sinus"] = sinus[0, 0]
                    tensors[f"{tag}.t_emb"] = emb[0, 0]
                    tensors[f"{tag}.lora"] = lora[0, 0]
                    tensors[f"{tag}.mod0"] = mod0[0, 0]
                    tensors[f"{tag}.mod_final"] = mod_final[0, 0]
                    tensors[f"{tag}.rope_cos"] = rope_cos
                    tensors[f"{tag}.rope_sin"] = rope_sin
                    tensors[f"{tag}.x_embed"] = x_emb.reshape(-1, d)
                    tensors[f"{tag}.block0"] = blk0.reshape(-1, d).float()
                tensors[f"{tag}.out"] = out[0, :, 0]
                rows[f"{tag}.out"] = stats(out)
                print(f"  depth {n_layers} case {ci} ({lat_h}x{lat_w}): "
                      f"ctx {tuple(ctx.shape)} out {tuple(out.shape)} "
                      f"|out| max {stats(out)[2]:.4f}")
        del model

    meta = {
        "checkpoint": os.path.basename(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "text_encoder": os.path.basename(args.text_encoder),
        "text_encoder_sha256": sha256(args.text_encoder),
        "config": json.dumps(cfg),
        "sampling_settings": json.dumps(comfy.supported_models.Anima.sampling_settings),
        "ref_layers": str(REF_LAYERS),
        "trend_layers": str(TREND_LAYERS),
        "adapter_layers": "full",
        "seed": str(SEED),
        "prompts": json.dumps(PROMPTS),
        "latents": json.dumps(LATENTS),
        "stats": json.dumps(rows),
    }
    comfy.utils.save_torch_file(tensors, args.out, metadata=meta)
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {args.out}  ({len(tensors)} tensors, {total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
