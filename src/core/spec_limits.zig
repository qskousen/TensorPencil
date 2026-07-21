//! Speculative-decoding size limits — the compile-time caps that both the
//! speculative-decode driver (`llm/spec.zig`) and the model backends need.
//!
//! These live here, at the core level, rather than in `llm/spec.zig` because
//! the model backends size their K/V and verify buffers from them: keeping the
//! constants below the model layer lets `models/*` depend on them downward
//! without pulling in the whole spec driver (which itself depends upward on the
//! generation engine + chat templating). `llm/spec.zig` re-exports both names,
//! so existing `spec.max_draft` / `spec.max_tree_nodes` references keep working.

/// Hard cap on drafted tokens per verify forward (draft buffers are
/// stack-allocated; `opts.spec_k` is clamped to this).
pub const max_draft = 16;

/// Hard cap on tree-verify nodes (root + drafted branches, LLM_PLAN.md M8);
/// `opts.tree_nodes` is clamped to this and backend tree buffers are sized for
/// it. Note that verify batches beyond `max_draft + 1` rows leave the grouped-
/// GEMV regime on the GPU backends (correct and lossless, but the GEMM path's
/// reduction order is no longer bitwise-identical to decode).
pub const max_tree_nodes = 64;
