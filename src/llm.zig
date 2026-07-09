//! LLM inference (tp-llm): chat templating, sampling, and the generation
//! loop over models.qwen3.CausalLM. See LLM_PLAN.md.

pub const chat = @import("llm/chat.zig");
pub const sample = @import("llm/sample.zig");
pub const engine = @import("llm/engine.zig");

test {
    _ = chat;
    _ = sample;
    _ = engine;
}
