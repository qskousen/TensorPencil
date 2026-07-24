//! tp_core — the foundational layer: pure data/primitive types with no
//! dependency on any compute backend, model, or the generation pipeline.
//! Everything above (ops, gpu, runtime, models, the umbrella) imports this
//! module by name (`@import("tp_core")`); nothing here imports upward.
//!
//! Contents: dtypes + tensors, checkpoint parsing (safetensors / GGUF) and
//! weight stores, block-quant dequant (`quants`, ggml-backed), the tokenizer,
//! the autoregressive K/V cache, the logits→token sampler, the diffusion
//! scheduler (`sampler`), Torch RNG, speculative-decode size limits, and the
//! profiling helper.

pub const dtype = @import("dtype.zig");
pub const tensor = @import("tensor.zig");
pub const quants = @import("quants.zig");
pub const quants_fixtures = @import("quants_fixtures.zig");
pub const safetensors = @import("safetensors.zig");
pub const gguf = @import("gguf.zig");
pub const weights = @import("weights.zig");
pub const torch_rng = @import("torch_rng.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const jinja = @import("jinja.zig");
pub const unicode_tables = @import("unicode_tables.zig");
pub const image = @import("image.zig");
pub const kv_cache = @import("kv_cache.zig");
pub const sample = @import("sample.zig");
pub const sampler = @import("sampler.zig");
pub const spec_limits = @import("spec_limits.zig");
pub const prof = @import("prof.zig");

test {
    _ = dtype;
    _ = tensor;
    _ = quants;
    _ = quants_fixtures;
    _ = safetensors;
    _ = gguf;
    _ = weights;
    _ = torch_rng;
    _ = tokenizer;
    _ = jinja;
    _ = unicode_tables;
    _ = image;
    _ = kv_cache;
    _ = sample;
    _ = sampler;
    _ = spec_limits;
    _ = prof;
}
