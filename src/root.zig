//! TensorPencil, a pure-Zig diffusion (text-to-image) inference engine.
//!
//! Public module root: everything consumers can reach is re-exported here.

const std = @import("std");

pub const init_defaults = @import("tp_core").init_defaults;
pub const dtype = @import("tp_core").dtype;
pub const tensor = @import("tp_core").tensor;
pub const safetensors = @import("tp_core").safetensors;
pub const gguf = @import("tp_core").gguf;
pub const quants = @import("tp_core").quants;
pub const weights = @import("tp_core").weights;
// Core primitives (pure-std, no device deps): the autoregressive K/V cache and the
// logits->token sampler. They live in tp_core so the GPU backend and every model can
// depend on them downward; also reachable as `llm.kv_cache` / `llm.sample`.
pub const kv_cache = @import("tp_core").kv_cache;
pub const sample = @import("tp_core").sample;
pub const ops = @import("tp_ops");
pub const tokenizer = @import("tp_core").tokenizer;
/// CLIP's BPE tokenizer and the weighted, multi-chunk `Prompt` the SD-family
/// conditioner is driven from, public because a caller composing the stages itself
/// needs `Prompt` to reach `clip_text.TextEncoder.encodePrompt`.
pub const clip_tokenizer = @import("tp_core").clip_tokenizer;
pub const image = @import("tp_core").image;
/// Waveform container I/O (WAV) and torchaudio-compatible rate conversion.
pub const audio = @import("tp_core").audio;
pub const models = @import("tp_models").models;
pub const embed = @import("embed.zig");
pub const sampler = @import("tp_core").sampler;
/// Which generator seeded noise comes from, ComfyUI's CPU torch RNG or the NVIDIA
/// Philox one A1111 draws with. Public because a consumer reproducing either
/// ecosystem's seeds needs to name the source, not just the seed.
pub const noise = @import("tp_core").noise;
pub const noise_curve = @import("tp_core").noise_curve;
pub const pipeline = @import("pipeline.zig");
pub const vram = @import("tp_runtime").vram;
/// CPU<->GPU layer-offload scheduling (shared by the CLI and the GUI, so the
/// residency rules are defined once, see `vram.resolve`).
pub const residency = @import("tp_runtime").residency;
pub const gpu = @import("tp_gpu");
pub const llm = @import("tp_models").llm;
pub const prof = @import("tp_core").prof;

pub const DType = dtype.DType;
pub const Shape = tensor.Shape;
pub const Tensor = tensor.Tensor;
pub const TensorInfo = tensor.TensorInfo;
pub const SafeTensors = safetensors.SafeTensors;
pub const TensorView = safetensors.TensorView;
pub const Gguf = gguf.Gguf;
pub const WeightStore = weights.WeightStore;

test {
    _ = dtype;
    _ = tensor;
    _ = safetensors;
    _ = gguf;
    _ = quants;
    _ = weights;
    _ = kv_cache;
    _ = sample;
    _ = ops;
    _ = tokenizer;
    _ = image;
    _ = models;
    _ = embed;
    _ = sampler;
    _ = pipeline;
    _ = vram;
    _ = gpu;
    _ = llm;
}
