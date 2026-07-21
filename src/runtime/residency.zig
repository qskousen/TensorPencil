//! Backend/model-agnostic CPU<->GPU layer-offload SCHEDULING for the CUDA
//! steppers. The dynamic-offload control flow (migrate-next, offload-until,
//! promote-back) was byte-identical across qwen35_cuda and gemma3_cuda; this
//! single-sources it, mirroring `transformer_gpu.decoderLayer`: the loop/order
//! lives here once, while the per-layer DEVICE work (moving a layer's KV /
//! weights / recurrent state, and the host-shadow state itself) stays on the
//! stepper `st`, which this drives through a small duck-typed contract:
//!
//!   st.split: ?Split          scheduling state. Field names are shared across
//!                             archs (the host-shadow fields differ and are only
//!                             touched by the per-model hooks below):
//!                               .dynamic: bool   .budget: u64
//!                               .order: []usize  .next: usize
//!   st.be                      the CUDA backend — `deviceUsed()` / `headroom()`
//!   st.migrateLayer(l) !void   move layer l device->host, freeing its VRAM
//!   st.promoteLayer(l) !void   bring layer l host->device, restoring its state
//!   st.promoteCost(l) usize    VRAM a promote of layer l needs (weights + the
//!                              KV it re-commits at capacity + slack)
//!
//! `st` is the `*CudaLM` pointer; the hooks are `pub` methods on it. A stepper
//! with no split (gemma4's resident MVP, qwen3's spec path) simply never calls
//! these — the `st.split == null` guards make them safe no-ops regardless.

const std = @import("std");

/// A point-in-time view of a stepper's CPU/GPU residency split + the card's free
/// VRAM, for the GUI's offload telemetry (logged whenever residency changes).
/// `n_cpu`/`n_layers` count the layers migrated to the host vs the total; the
/// byte figures are the live device usage and free VRAM in MiB.
pub const Snapshot = struct {
    n_cpu: usize,
    n_layers: usize,
    device_mib: u64,
    free_mib: u64,
};

/// Snapshot `st`'s residency. Must run on the thread that bound this model's
/// CUDA context (memGetInfo/deviceUsed read the current context).
pub fn snapshot(st: anytype) Snapshot {
    return .{
        .n_cpu = if (st.split) |sp| sp.n_cpu else 0,
        .n_layers = st.cfg.n_layers,
        .device_mib = st.be.deviceUsed() >> 20,
        .free_mib = st.be.ctx.memGetInfo().free >> 20,
    };
}

/// Migrate the next layer in the offload order to the host (dynamic mode).
/// Returns false when nothing is left to migrate.
pub fn migrateNext(st: anytype) !bool {
    const sp = &st.split.?;
    if (sp.next >= sp.order.len) return false;
    const l = sp.order[sp.next];
    sp.next += 1;
    try st.migrateLayer(l);
    return true;
}

/// Migrate layers to the host until `@min(budget - deviceUsed, headroom)`
/// reaches `needed_free` bytes, or nothing is left. No-op without a dynamic
/// split. Fixed-target variant used by the VRAM coordinator to free room for a
/// resident image model. (`ensureCapacity` keeps its own loop, whose target
/// shrinks per iteration as live slots drop — a fixed target can't express that.)
pub fn offloadUntilFree(st: anytype, needed_free: u64) !void {
    if (st.split == null) return;
    const sp = &st.split.?;
    if (!sp.dynamic) return;
    while (true) {
        const free = @min(sp.budget -| st.be.deviceUsed(), st.be.headroom());
        if (free >= needed_free) break;
        if (!(try migrateNext(st))) break; // nothing left
    }
}

/// Migrate layers until the LLM's actual total device usage is at or under
/// `target` bytes (the GUI's `balanced` mode: settle the LLM to its share only
/// when an image model contends). Live `deviceUsed()`, one-way + idempotent.
/// No-op without a dynamic split.
pub fn offloadToBudget(st: anytype, target: u64) !void {
    if (st.split == null or target == 0) return;
    const sp = &st.split.?;
    if (!sp.dynamic) return;
    while (st.be.deviceUsed() > target) {
        if (!(try migrateNext(st))) break; // nothing left to migrate
    }
}

/// Settle device residency to `target` bytes and set it as the ongoing KV-growth
/// ceiling — the enactment primitive the cross-model VRAM arbiter drives each
/// model to (`vram.Participant.applyBudget`). Arms the dynamic split if it
/// isn't already, records `target` as the ceiling honored while KV grows, then
/// migrates layers host-ward (currently over budget) or promotes them back
/// (under budget). Idempotent: a satisfied target settles nothing. A no-op on
/// `target == 0`.
///
/// Backend residency mutation, so the caller MUST be on the thread that bound
/// this model's CUDA context (the model's own worker at a safe boundary, or the
/// arbiter thread while the model is idle). This is a faithful lift of the GUI's
/// former `settleLlm` core; the thread-bind + telemetry stay at the call site.
pub fn settleTo(st: anytype, target: u64) !void {
    if (target == 0) return;
    if (st.split == null) _ = try st.autoOffload(target);
    if (st.split) |*sp| sp.budget = target;
    if (st.be.deviceUsed() > target)
        try st.offloadToBudget(target)
    else
        _ = try st.promoteLayers(target);
}

/// Migrate CPU layers back onto the GPU (LIFO by offload order), stopping before
/// the next would overflow `budget` — so the caller (VRAM coordinator, after
/// image generation) reclaims LLM residency while leaving room for whatever else
/// stays resident. Keeps the split armed (offload can fire again). Returns the
/// number promoted; 0 without a split.
pub fn promoteBack(st: anytype, budget: u64) !usize {
    if (st.split == null) return 0;
    const sp = &st.split.?;
    var promoted: usize = 0;
    while (sp.next > 0) {
        const l = sp.order[sp.next - 1];
        const cost = st.promoteCost(l);
        const free = @min(budget -| st.be.deviceUsed(), st.be.headroom());
        if (free < cost) break;
        try st.promoteLayer(l);
        sp.next -= 1;
        promoted += 1;
    }
    return promoted;
}
