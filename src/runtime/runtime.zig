//! tp_runtime, the offload / scheduling tier: VRAM budgeting (`vram`, the
//! Arbiter), the CPU/GPU layer-residency planner (`residency`) and the stepper
//! boundary hook the two are enacted through (`boundary`). Pure logic
//! (std-only): these compute budgets and offload plans; the actual device
//! allocations/copies happen in the model backends that consume the plans.
//! Sits above tp_gpu conceptually but has no compile dependency on it.

pub const vram = @import("vram.zig");
pub const residency = @import("residency.zig");
pub const boundary = @import("boundary.zig");

test {
    _ = vram;
    _ = residency;
    _ = boundary;
}
