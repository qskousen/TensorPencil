# Dump torch.randn CPU fixtures for the Zig torch-compatible RNG
# (src/torch_rng.zig). Run with ComfyUI's venv:
#   ~/genai/comfyui/nvenv/bin/python tools/dump_randn.py
#
# ComfyUI's prepare_noise is torch.manual_seed(seed) + torch.randn(shape,
# device="cpu", dtype=float32); these fixtures pin that path bit-for-bit,
# including the >= 16-element normal_fill blocks and the overlapping tail
# (sizes not divisible by 16 regenerate the last 16 values).

import torch


def dump(seed, shape, path):
    g = torch.manual_seed(seed)
    t = torch.randn(shape, generator=g, device="cpu", dtype=torch.float32)
    t.numpy().tofile(path)
    print(f"{path}: seed={seed} shape={tuple(shape)} first={t.flatten()[0].item():.9g}")


# Latent-like shape (numel % 16 == 0, the only case the pipeline hits).
dump(42, (1, 16, 8, 8), "testdata/randn_42_1024.bin")
# Overlapping-tail case (numel % 16 != 0).
dump(7, (40,), "testdata/randn_7_40.bin")
# The comparison workflow's actual latent, truncated head for a spot check.
g = torch.manual_seed(80085)
t = torch.randn((1, 16, 210, 140), generator=g, device="cpu", dtype=torch.float32)
t.flatten()[:1024].numpy().tofile("testdata/randn_80085_head1024.bin")
print(f"testdata/randn_80085_head1024.bin: first={t.flatten()[0].item():.9g}")
