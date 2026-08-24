//! tp_core, the foundational layer: pure data/primitive types with no
//! dependency on any compute backend, model, or the generation pipeline.
//! Everything above (ops, gpu, runtime, models, the umbrella) imports this
//! module by name (`@import("tp_core")`); nothing here imports upward.
//!
//! Contents: dtypes + tensors, checkpoint parsing (safetensors / GGUF) and
//! weight stores, block-quant dequant (`quants`, ggml-backed), the tokenizer,
//! the autoregressive K/V cache, the logits->token sampler, the diffusion
//! scheduler (`sampler`), Torch RNG, speculative-decode size limits, and the
//! profiling helper.

pub const init_defaults = @import("init_defaults.zig");
pub const dtype = @import("dtype.zig");
pub const tensor = @import("tensor.zig");
pub const quants = @import("quants.zig");
pub const quants_fixtures = @import("quants_fixtures.zig");
pub const safetensors = @import("safetensors.zig");
pub const gguf = @import("gguf.zig");
pub const weights = @import("weights.zig");
pub const torch_rng = @import("torch_rng.zig");
/// NVIDIA's Philox, the generator A1111 draws its noise from, unlike ComfyUI's CPU one.
pub const philox_rng = @import("philox_rng.zig");
/// Which of the two the noise comes from; both consumers dispatch through here.
pub const noise = @import("noise.zig");
pub const noise_curve = @import("noise_curve.zig");
/// `numpy.random.SeedSequence`, for the Brownian-tree noise sampler below.
pub const seed_seq = @import("seed_seq.zig");
/// torchsde's Brownian tree, the noise source ComfyUI's SDE samplers draw from.
pub const brownian = @import("brownian.zig");
pub const tokenizer = @import("tokenizer.zig");
/// CLIP BPE, the SD family's prompt tokenizer (see that file for why it is not a
/// variant of `tokenizer.zig`).
pub const clip_tokenizer = @import("clip_tokenizer.zig");
/// ComfyUI's `(a:1.2)` emphasis parser, shared by every tokenizer that consumes it
/// (`clip_tokenizer` for the SD family, `t5_tokenizer` for Anima).
pub const prompt_weights = @import("prompt_weights.zig");
/// T5 SentencePiece-Unigram, Anima's SECOND prompt tokenizer, whose ids index the
/// `llm_adapter`'s own embedding (there is no T5 model here).
pub const t5_tokenizer = @import("t5_tokenizer.zig");
/// AUTOMATIC1111 prompt dialect (emphasis + per-step scheduling), alongside
/// `clip_tokenizer`'s ComfyUI one.
pub const prompt_a1111 = @import("prompt_a1111.zig");
pub const jinja = @import("jinja.zig");
pub const unicode_tables = @import("unicode_tables.zig");
pub const image = @import("image.zig");
pub const kv_cache = @import("kv_cache.zig");
pub const sample = @import("sample.zig");
/// Diffusion sigma SCHEDULES (where the steps go), all nine ComfyUI schedulers.
pub const schedule = @import("schedule.zig");
/// Diffusion SAMPLERS (how to step), Euler and DPM++ 2M SDE.
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
    _ = philox_rng;
    _ = noise;
    _ = noise_curve;
    _ = seed_seq;
    _ = brownian;
    _ = tokenizer;
    _ = clip_tokenizer;
    _ = prompt_weights;
    _ = t5_tokenizer;
    _ = prompt_a1111;
    _ = jinja;
    _ = unicode_tables;
    _ = image;
    _ = kv_cache;
    _ = sample;
    _ = schedule;
    _ = sampler;
    _ = spec_limits;
    _ = prof;
}
