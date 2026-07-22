//! TensorPencil — a pure-Zig diffusion (text-to-image) inference engine.
//!
//! Public module root: everything consumers can reach is re-exported here.

const std = @import("std");

pub const dtype = @import("tp_core").dtype;
pub const tensor = @import("tp_core").tensor;
pub const safetensors = @import("tp_core").safetensors;
pub const gguf = @import("tp_core").gguf;
pub const quants = @import("tp_core").quants;
pub const weights = @import("tp_core").weights;
// Core primitives that used to live under llm/ (pure-std, no device deps):
// the autoregressive K/V cache and the logits→token sampler. Relocated so the
// GPU backend and every model can depend on them downward (also still reachable
// as `llm.kv_cache` / `llm.sample` for backward compatibility).
pub const kv_cache = @import("tp_core").kv_cache;
pub const sample = @import("tp_core").sample;
pub const ops = @import("tp_ops");
pub const tokenizer = @import("tp_core").tokenizer;
pub const image = @import("tp_core").image;
pub const models = @import("tp_models").models;
pub const embed = @import("embed.zig");
pub const sampler = @import("tp_core").sampler;
pub const pipeline = @import("pipeline.zig");
pub const vram = @import("tp_runtime").vram;
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
