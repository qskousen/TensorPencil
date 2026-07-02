"""Generate Zig-ready reference values for src/ops tests using torch.

Run with ComfyUI's venv python:
  ~/genai/comfyui/venv/bin/python tools/gen_op_fixtures.py
Copy the printed arrays into the corresponding Zig test blocks.
"""
import math
import torch

torch.manual_seed(0)

def zarr(name, t):
    vals = ", ".join(f"{v:.9g}" for v in t.flatten().tolist())
    print(f"const {name} = [_]f32{{ {vals} }};")

# --- activations ---
x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 0.1, -2.2])
zarr("act_in", x)
zarr("gelu_tanh_out", torch.nn.functional.gelu(x, approximate="tanh"))
zarr("silu_out", torch.nn.functional.silu(x))
zarr("sigmoid_out", torch.sigmoid(x))

# --- rmsnorm, (1+scale) convention, eps 1e-5, f32 (krea2 RMSNorm) ---
xr = torch.tensor([[0.5, -1.5, 2.0, 0.25], [3.0, 0.0, -0.125, 1.0]])
scale = torch.tensor([0.1, -0.2, 0.0, 0.5])
out = torch.nn.functional.rms_norm(xr, (4,), weight=scale + 1.0, eps=1e-5)
zarr("rmsnorm_x", xr); zarr("rmsnorm_scale", scale); zarr("rmsnorm_out", out)
# plain-weight variant (qwen convention), eps 1e-6
out2 = torch.nn.functional.rms_norm(xr, (4,), weight=scale + 1.0, eps=1e-6)
zarr("rmsnorm_plain_out", out2)  # same weight values passed directly

# --- flux rope: dim 8, theta 1000, positions [0,1,7], one 8-dim vector per pos ---
from einops import rearrange
def rope(pos, dim, theta):
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float32), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

pos = torch.tensor([[0.0, 1.0, 7.0]])
freqs = rope(pos, 8, 1000)  # [1, 3, 4, 2, 2]
q = torch.arange(24, dtype=torch.float32).reshape(1, 1, 3, 8) / 10.0 - 1.0  # [B,H,L,D]
q_ = q.reshape(*q.shape[:-1], -1, 1, 2)
fc = freqs.unsqueeze(1)  # [1,1,3,4,2,2]
q_out = (fc[..., 0] * q_[..., 0] + fc[..., 1] * q_[..., 1]).reshape(q.shape)
zarr("rope_q_in", q); zarr("rope_q_out", q_out)

# --- timestep embedding: dim 8, t = [0.25, 1.0] ---
def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
zarr("temb_out", timestep_embedding(torch.tensor([0.25, 1.0]), 8))

# --- GQA attention: seq 3, heads 2, kv_heads 1, head_dim 4 ---
qa = torch.randn(1, 2, 3, 4)  # [B,H,L,D]
ka = torch.randn(1, 1, 3, 4)
va = torch.randn(1, 1, 3, 4)
k_r = ka.repeat_interleave(2, dim=1)
v_r = va.repeat_interleave(2, dim=1)
out_a = torch.nn.functional.scaled_dot_product_attention(qa, k_r, v_r)
# flatten to [L, H*D] row-major activations like our engine uses
zarr("attn_q", qa.permute(0, 2, 1, 3).reshape(3, 8))
zarr("attn_k", ka.permute(0, 2, 1, 3).reshape(3, 4))
zarr("attn_v", va.permute(0, 2, 1, 3).reshape(3, 4))
zarr("attn_out", out_a.permute(0, 2, 1, 3).reshape(3, 8))
# causal variant
out_c = torch.nn.functional.scaled_dot_product_attention(qa, k_r, v_r, is_causal=True)
zarr("attn_causal_out", out_c.permute(0, 2, 1, 3).reshape(3, 8))
