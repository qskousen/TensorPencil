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
const vram = @import("vram.zig");

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

/// Scheduling margin each stepper's `promoteCost` adds on top of a layer's real
/// device bytes, so a promote that *just* fits doesn't land the card at the
/// physical edge. Single-sourced here because `demand` has to subtract it back
/// out: it wants a FOOTPRINT, not a promote cost.
/// Untyped so it coerces into both the steppers' `usize` promote costs and the
/// `u64` byte budgets here.
pub const promote_slack = 64 << 20;

/// Device residency `st` is ACTUALLY holding right now, itemized.
///
///   weights  the backend's pinned-weight counter. LLM weights are always pinned
///            (`pinAllWeights`), so this is exactly "parameters on the device".
///   kv       summed over the layers whose K/V still lives on the device.
///   scratch  the remainder of `deviceUsed()` — activations, logits, RoPE tables,
///            dequant staging. Derived rather than enumerated so it can never
///            silently omit a buffer: anything the allocator counted lands here.
pub fn measured(st: anytype) vram.Bytes {
    const w = st.be.pinnedWeightBytes();
    var kv: u64 = 0;
    for (0..st.cfg.n_layers) |l| {
        if (!onDevice(st, l)) continue;
        kv += layerKvBytes(st, l);
    }
    const used = st.be.deviceUsed();
    return .{ .weights = w, .kv = kv, .scratch = used -| w -| kv };
}

/// Device residency `st` WOULD hold with every layer back on the GPU — its full
/// resident footprint, i.e. `vram.Participant.need`. Equal to `measured` when
/// nothing is offloaded and the model has run a batched forward.
///
/// Built by taking the measurement and replacing the two terms that residency
/// changes — all weights instead of the uploaded ones, every layer's KV instead
/// of the resident layers' — then adding the one allocation that does not exist
/// yet on a cold model. Anchoring on the measurement is what keeps this honest:
/// `scratch` carries through unmodified, so the activation buffers, logits and
/// RoPE tables are MEASURED rather than guessed at. An earlier version summed
/// per-layer costs from scratch and silently omitted all of them, which (since
/// the arbiter turns this into a residency ceiling) capped the model below its
/// own footprint and stranded layers on the host with the card half free.
pub fn need(st: anytype) vram.Bytes {
    var b = measured(st);
    b.weights = nonLayerBytes(st);
    b.kv = 0;
    for (0..st.cfg.n_layers) |l| {
        b.weights += st.layerWeightBytes(l);
        b.kv += layerKvBytes(st, l);
    }
    b.scratch += firstForwardScratch(st);
    return b;
}

/// Full resident footprint as a single figure (`vram.Participant.demand`).
pub fn demand(st: anytype) u64 {
    return need(st).total();
}

/// Is layer `l`'s K/V still on the device? (Everything is, without a split.)
fn onDevice(st: anytype, l: usize) bool {
    const sp = if (st.split) |*s| s else return true;
    const n_host = @min(sp.next, sp.order.len);
    for (sp.order[0..n_host]) |h| if (h == l) return false;
    return true;
}

/// Device K/V one layer commits at the current capacity. `promoteCost` is that
/// plus the layer's weights plus scheduling slack, so back the other two out
/// rather than adding a second per-arch hook that could drift from it.
fn layerKvBytes(st: anytype, l: usize) u64 {
    return @as(u64, st.promoteCost(l)) -| promote_slack -| st.layerWeightBytes(l);
}

/// Dequant staging that a batched forward may allocate: `opMatmulQuant` expands
/// the widest weight it is handed to f16 and converts the activation block
/// alongside it. Zero once the model is WARM — by then whatever it really
/// allocated is already inside `measured.scratch`, and adding a prediction on top
/// would inflate `need` past reality.
///
/// That distinction matters because the estimate is necessarily pessimistic: it
/// assumes the widest layer linear goes through the dequant path, but a model
/// whose weights are all q4_k now takes the MMQ path instead and never allocates
/// the f16 staging at all. Measured on gemma4-31B, predicting it unconditionally
/// put `need` 225 MiB above actual — safe direction, but it is 225 MiB of headroom
/// spent on a buffer that does not exist, and headroom is what stops layers being
/// offloaded. So: predict only while cold, measure once warm.
///
/// The widest layer linear is the MLP in every arch here (`intermediate >= qDim`
/// for qwen3/qwen35/gemma3/gemma4). The LM head is wider on large-vocab models but
/// never uses this path — it is a GEMV (`opGemvQuant`), which streams the weight.
fn firstForwardScratch(st: anytype) u64 {
    if (st.be.pinnedWeightBytes() != 0) return 0; // warm: the measurement is the truth
    const c = st.cfg;
    const w16 = @as(u64, c.intermediate) * c.hidden * 2;
    const a16 = @as(u64, prefill_rows_hint) * c.intermediate * 2;
    return (w16 + a16) -| st.be.dequantScratchBytes();
}

/// Rows the activation half of the dequant scratch is sized for. The steppers
/// prefill in chunks of this order; it only scales the smaller of the two terms,
/// so being a chunk out costs a few MiB of estimate, not a layer.
const prefill_rows_hint = 256;

/// Device bytes a fully-resident model holds that are NOT per-layer: the LM head
/// and the token embeddings (both land in the backend weight cache on first use).
/// `promoteCost` covers layers only, so without this the COLD demand bound reports
/// less than the model costs the moment it IS fully resident.
///
/// That gap is load-bearing, not cosmetic: the uncontended arm of `Arbiter.plan`
/// hands the LLM exactly its demand, so an under-estimate becomes a hard residency
/// ceiling and the model offloads layers to the host to get under a number below
/// what it actually needs — with the card's real budget unspent. Measured on a
/// 3090 with nothing else resident (qwen3-32B, 64 layers): demand 19860 MiB vs a
/// 20829 MiB budget and 20832 MiB of true full residency, so 3 layers were pushed
/// to the host and 1.5 GiB sat idle. The 972 MiB shortfall is exactly this term.
///
/// Still an estimate: activation/logits scratch is not included (it has no static
/// size), so this remains a lower bound when cold. Once the model is warm the
/// `warm` bound in `demand` supersedes it anyway — `used` already counts
/// everything, and algebraically `all_layers + nonLayer == warm` there.
///
/// Steppers without the hook keep the old behaviour rather than silently mis-report.
fn nonLayerBytes(st: anytype) u64 {
    const M = @typeInfo(@TypeOf(st)).pointer.child;
    if (!@hasDecl(M, "nonLayerDeviceBytes")) return 0;
    return st.nonLayerDeviceBytes();
}

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

/// Resolve a CLI-style absolute `--vram-budget` (bytes; 0 = unset) into a device
/// residency cap, through the SAME `vram.resolve` rule the GUI meter uses — so
/// "limit" means one thing across both frontends instead of two formulas that can
/// drift. Returns 0 when unset, which every split planner already reads as "no
/// budget, no offload".
///
/// The CLI has no NVML, so `foreign` degrades to the residual
/// `device_used - our_tracked`; that folds our own untracked bytes into the
/// reserve, which is conservative in the right direction. The gain over passing
/// the flag straight through is that a `--vram-budget` larger than the card can
/// physically hold is now clamped instead of being taken at face value.
pub fn resolveBudget(st: anytype, vram_budget: u64) u64 {
    if (vram_budget == 0) return 0;
    const mi = st.be.ctx.memGetInfo();
    if (mi.total == 0) return vram_budget; // no card info: honour the flag as given
    const tracked = st.be.deviceUsed();
    const r = vram.resolve(.{ .ours_bytes = vram_budget }, .{
        .total = mi.total,
        .foreign = null,
        .ours_tracked = tracked,
        .device_used = mi.total -| mi.free,
    }, 0);
    return r.tracked;
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

/// Free VRAM a promote must leave untouched, beyond the per-layer cost: the
/// per-token activation/logits allocations plus KV-growth churn need live
/// headroom, and a promote that lands the card at the physical edge starts a
/// per-token OOM→offload→promote thrash cycle. Promoting late is harmless
/// (a settle retries next pump); promoting into the edge is not.
const promote_reserve: u64 = 256 << 20;

/// Free VRAM a promote must leave, for THIS model. The flat `promote_reserve`
/// above is only a floor: a forward pass allocates transient buffers whose size
/// depends on the architecture and batch (measured on gemma4-31B: a single
/// 1102 MiB request during prefill), and filling the ceiling to within 256 MiB of
/// the card guarantees the next one OOMs — which offloads layers, which is the
/// churn this path exists to avoid. The backend reports its own high-water, so
/// this is measured rather than guessed and needs no per-arch estimate.
fn promoteReserve(st: anytype) u64 {
    return @max(promote_reserve, st.be.largestAllocation());
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
    // A layer's WEIGHTS promote lazily (promoteLayer re-creates only its KV;
    // the weights re-cache pinned on the next forward), so deviceUsed() does
    // not yet include them. Carry that deferred cost across the loop —
    // without it one settle promoted a dozen layers, the next forward's
    // weight uploads OOM'd, the retry offloaded them all back, and the next
    // settle promoted again: a full per-token PCIe thrash cycle.
    var deferred: u64 = 0;
    while (sp.next > 0) {
        const l = sp.order[sp.next - 1];
        const cost = st.promoteCost(l);
        const used = st.be.deviceUsed() + deferred;
        const free = @min(budget -| used, st.be.headroom() -| deferred);
        if (free < cost + promoteReserve(st)) break;
        const before = st.be.deviceUsed();
        try st.promoteLayer(l);
        // Whatever promoteLayer did NOT allocate now (its weight bytes, part
        // of `cost`) arrives at the next forward; count it as spent already.
        deferred += cost -| (st.be.deviceUsed() -| before);
        sp.next -= 1;
        promoted += 1;
    }
    return promoted;
}
