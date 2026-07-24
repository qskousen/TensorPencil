//! LLM inference (tp-llm): chat templating, sampling, the generation loop
//! over models.qwen3.CausalLM, and speculative decoding. See LLM_PLAN.md.

pub const chat = @import("llm/chat.zig");
pub const chat_template = @import("llm/chat_template.zig");
pub const repl = @import("llm/repl.zig");
pub const sample = @import("tp_core").sample;
pub const engine = @import("llm/engine.zig");
pub const spec = @import("llm/spec.zig");
pub const kv_cache = @import("tp_core").kv_cache;
pub const session = @import("llm/session.zig");

test {
    _ = chat;
    _ = chat_template;
    _ = repl;
    _ = sample;
    _ = engine;
    _ = spec;
    _ = kv_cache;
    _ = session;
}
