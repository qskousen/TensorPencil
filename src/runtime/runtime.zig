//! tp_runtime — the offload / scheduling tier: VRAM budgeting (`vram`, the
//! Arbiter) and the CPU/GPU layer-residency planner (`residency`). Pure logic
//! (std-only): these compute budgets and offload plans; the actual device
//! allocations/copies happen in the model backends that consume the plans.
//! Sits above tp_gpu conceptually but has no compile dependency on it.

pub const vram = @import("vram.zig");
pub const residency = @import("residency.zig");

test {
    _ = vram;
    _ = residency;
}
