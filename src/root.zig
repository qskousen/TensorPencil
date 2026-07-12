//! TensorPencil — a pure-Zig diffusion (text-to-image) inference engine.
//!
//! Public module root: everything consumers can reach is re-exported here.

const std = @import("std");

pub const dtype = @import("dtype.zig");
pub const tensor = @import("tensor.zig");
pub const safetensors = @import("safetensors.zig");
pub const gguf = @import("gguf.zig");
pub const quants = @import("quants.zig");
pub const weights = @import("weights.zig");
pub const ops = @import("ops.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const image = @import("image.zig");
pub const models = @import("models.zig");
pub const sampler = @import("sampler.zig");
pub const pipeline = @import("pipeline.zig");
pub const gpu = @import("gpu.zig");
pub const llm = @import("llm.zig");
pub const prof = @import("prof.zig");

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
    _ = ops;
    _ = tokenizer;
    _ = image;
    _ = models;
    _ = sampler;
    _ = pipeline;
    _ = gpu;
    _ = llm;
}
