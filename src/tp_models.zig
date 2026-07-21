//! tp_models — the model tier: neural-net architectures plus the LLM generation
//! machinery that drives them. Depends on tp_core, tp_ops, tp_gpu, tp_runtime.
//!
//! `models` holds the architectures (text encoders, DiT, VAE, qwen/gemma LLMs,
//! ViT towers, the EAGLE-3 drafter). `llm` holds the generation orchestration
//! that operates over any model via duck typing — chat templating, the decode
//! `engine`, speculative decoding (`spec`), the REPL, and the session wrapper.
//! They live in one module because they are mutually recursive (spec drafts for
//! a concrete model; models' integration tests drive the engine), and neither
//! depends on the diffusion pipeline (the umbrella) above.
//!
//! `test_gate` is the integration-suite gate used by the tests in this tier.

pub const models = @import("models.zig");
pub const llm = @import("llm.zig");
pub const test_gate = @import("test_gate.zig");

test {
    _ = models;
    _ = llm;
    _ = test_gate;
}
