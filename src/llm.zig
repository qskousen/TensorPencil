//! LLM inference (tp-llm): chat templating, sampling, the generation loop
//! over models.qwen3.CausalLM, and speculative decoding. See LLM_PLAN.md.

pub const chat = @import("llm/chat.zig");
pub const repl = @import("llm/repl.zig");
pub const sample = @import("llm/sample.zig");
pub const engine = @import("llm/engine.zig");
pub const spec = @import("llm/spec.zig");

test {
    _ = chat;
    _ = repl;
    _ = sample;
    _ = engine;
    _ = spec;
}
