"""Per-stage Wan VAE decode dumps for debugging parity (T=1 path).

Writes testdata/vae_stage_<name>.bin, planar f32 [c][h][w], for the same seed-7
8x8 latent as dump_vae_fixture.py.
"""
import sys, types, torch
sys.path.insert(0, "/home/qt/genai/comfyui")
sys.argv = [sys.argv[0]]
_a = types.ModuleType("comfy_aimdo"); sys.modules["comfy_aimdo"] = _a
for s in ("host_buffer", "model_vbar", "torch", "vram_buffer"):
    m = types.ModuleType(f"comfy_aimdo.{s}"); sys.modules[f"comfy_aimdo.{s}"] = m; setattr(_a, s, m)
import comfy.sd, comfy.utils

sd = comfy.utils.load_torch_file("/dump/projects/zig/TensorPencil/models/vae/krea2RealVae_v10.safetensors")
vae = comfy.sd.VAE(sd=sd, dtype=torch.float32)
model = vae.first_stage_model.float().eval()

names = {}
names[id(model.conv2)] = "postquant"
names[id(model.decoder.conv1)] = "convin"
for i, m in enumerate(model.decoder.middle):
    names[id(m)] = f"mid{i}"
for i, m in enumerate(model.decoder.upsamples):
    names[id(m)] = f"up{i}"

def hook(mod, args, out):
    name = names[id(mod)]
    t = out[0] if isinstance(out, (list, tuple)) else out
    t = t.detach()
    if t.dim() == 5:
        t = t[0, :, 0]
    t.contiguous().numpy().tofile(f"/dump/projects/zig/TensorPencil/testdata/vae_stage_{name}.bin")
    print(name, tuple(t.shape))

for m in list(names_mods := []) : pass
for mod_id, name in list(names.items()):
    pass
for m in [model.conv2, model.decoder.conv1, *model.decoder.middle, *model.decoder.upsamples]:
    m.register_forward_hook(hook)

torch.manual_seed(7)
z = torch.randn(1, 16, 1, 8, 8, dtype=torch.float32)
with torch.no_grad():
    model.decode(z)
print("done")
