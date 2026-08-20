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
//!   st.be                      the CUDA backend: `deviceUsed()` / `headroom()`,
//!                              plus `pinnedWeightBytes()` /
//!                              `evictableWeightBytes()` for `splitHostCount`
//!   st.migrateLayer(l) !void   move layer l device->host, freeing its VRAM
//!   st.promoteLayer(l) !void   bring layer l host->device, restoring its state
//!   st.promoteCost(l) usize    VRAM a promote of layer l needs (weights + the
//!                              KV it re-commits at capacity + slack)
//!
//! `st` is the `*CudaLM` pointer; the hooks are `pub` methods on it. A stepper
//! with no split (gemma4's resident MVP, qwen3's spec path) simply never calls
//! these, the `st.split == null` guards make them safe no-ops regardless.

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
///   scratch  the remainder of `deviceUsed()`, activations, logits, RoPE tables,
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

/// Device residency `st` WOULD hold with every layer back on the GPU, its full
/// resident footprint, i.e. `vram.Participant.need`. Equal to `measured` when
/// nothing is offloaded and the model has run a batched forward.
///
/// Built by taking the measurement and replacing the two terms that residency
/// changes, all weights instead of the uploaded ones, every layer's KV instead
/// of the resident layers', then adding the one allocation that does not exist
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

/// Layers a fresh split must place on the HOST for the model to fit under
/// `budget`, taken from the front of `order` (first to leave, first). `per[l]`
/// is layer `l`'s device weight bytes. `reserve` is what must stay free for the
/// parameters that are NOT layer weights: the LM head for a dynamic split,
/// which migrates more layers later as the KV grows, and additionally the KV at
/// capacity plus slack for a static one, which cannot.
///
/// The whole difficulty is which device bytes the budget has already been spent
/// on. `deviceUsed()` includes the layer weights the loader has already
/// uploaded, and those are exactly what this is placing, so charging them to the
/// budget as well leaves almost nothing for the weights themselves and nearly
/// every layer plans onto the host. That split then costs the prefill and buys
/// no memory at all, because a host layer's weights are reclaimed lazily and so
/// sit in VRAM regardless. Netting the cached weights back out is what compares
/// `budget` against the same footprint `per` describes.
///
/// A `dynamic` split additionally leaves `promote_reserve` free for the first
/// forward's transient buffers. Its own `reserve` covers only parameters, so
/// without this the plan packs the card to within a few hundred MiB, the
/// forward's dequant staging has nowhere to go, and the OOM handler dumps a
/// batch of layers to the host: more offloaded than a plan that had left room,
/// and chosen reactively rather than by the offload order. Static splits keep
/// their own generous slack instead.
///
/// The flat floor, NOT `promoteReserve`: that maxes in `largestAllocation`,
/// which at load time is the LM head's upload (measured 1102 MiB on a 31B), a
/// buffer that is HELD rather than re-requested per forward. Reserving it costs
/// three extra layers for a transient no forward asks for. `promoteBack` reads
/// the high-water later, once a real forward has taught it what one costs.
pub fn splitHostCount(st: anytype, budget: u64, reserve: u64, dynamic: bool, per: []const usize, order: []const usize) usize {
    var total_weight: u64 = 0;
    for (per) |b| total_weight += b;

    const used = st.be.deviceUsed();
    // Device bytes that are not model parameters: committed KV, activation
    // scratch, RoPE tables. The parameters come back as `total_weight` (layers)
    // and `reserve` (the head), so each is counted once.
    const other = used -| st.be.pinnedWeightBytes() -| st.be.evictableWeightBytes();
    // The plan must respect the card's LIVE free VRAM, not just the abstract
    // budget: other processes may hold a chunk of the card. A plan the card
    // cannot satisfy places everything resident and then faults at the first
    // prefill, where weight uploads and lazy PTX JIT collide at zero free.
    // headroom() already keeps a 10% margin.
    const ceiling = @min(budget, used + st.be.headroom());
    const transient: u64 = if (dynamic) promote_reserve else 0;
    const weight_budget = ceiling -| other -| reserve -| transient;

    var gpu_weight = total_weight;
    var n_cpu: usize = 0;
    while (gpu_weight > weight_budget and n_cpu < order.len) {
        gpu_weight -= per[order[n_cpu]];
        n_cpu += 1;
    }
    if (n_cpu != 0)
        std.log.info("[vram] split plan: {d}/{d} layers to the host · layer weights {d} MiB vs {d} MiB (ceiling {d} - other {d} - reserve {d} - transient {d})", .{
            n_cpu,         order.len,   total_weight >> 20, weight_budget >> 20,
            ceiling >> 20, other >> 20, reserve >> 20,      transient >> 20,
        });
    return n_cpu;
}

/// Layer weight bytes living on the HOST, i.e. the parameters this model is
/// running on the CPU instead of the GPU. 0 when nothing is offloaded.
///
/// Weights only, not the K/V those layers carry with them: this is the figure the
/// status bar reports as the cost of the split, and a layer's K/V grows with the
/// conversation, which would make the number drift while the residency it is
/// describing sits still.
pub fn hostWeightBytes(st: anytype) u64 {
    if (st.split == null) return 0;
    var b: u64 = 0;
    for (0..st.cfg.n_layers) |l| {
        if (onDevice(st, l)) continue;
        b += st.layerWeightBytes(l);
    }
    return b;
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
/// alongside it. Zero once the model is WARM, by then whatever it really
/// allocated is already inside `measured.scratch`, and adding a prediction on top
/// would inflate `need` past reality.
///
/// That distinction matters because the estimate is necessarily pessimistic: it
/// assumes the widest layer linear goes through the dequant path, but a model
/// whose weights are all q4_k now takes the MMQ path instead and never allocates
/// the f16 staging at all. Measured on gemma4-31B, predicting it unconditionally
/// put `need` 225 MiB above actual, safe direction, but it is 225 MiB of headroom
/// spent on a buffer that does not exist, and headroom is what stops layers being
/// offloaded. So: predict only while cold, measure once warm.
///
/// The widest layer linear is the MLP in every arch here (`intermediate >= qDim`
/// for qwen3/qwen35/gemma3/gemma4). The LM head is wider on large-vocab models but
/// never uses this path, it is a GEMV (`opGemvQuant`), which streams the weight.
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
/// what it actually needs, with the card's real budget unspent. Measured on a
/// 3090 with nothing else resident (qwen3-32B, 64 layers): demand 19860 MiB vs a
/// 20829 MiB budget and 20832 MiB of true full residency, so 3 layers were pushed
/// to the host and 1.5 GiB sat idle. The 972 MiB shortfall is exactly this term.
///
/// Still an estimate: activation/logits scratch is not included (it has no static
/// size), so this remains a lower bound when cold. Once the model is warm the
/// `warm` bound in `demand` supersedes it anyway, `used` already counts
/// everything, and algebraically `all_layers + nonLayer == warm` there.
///
/// Steppers without the hook keep the old behaviour rather than silently mis-report.
fn nonLayerBytes(st: anytype) u64 {
    const M = @typeInfo(@TypeOf(st)).pointer.child;
    if (!@hasDecl(M, "nonLayerDeviceBytes")) return 0;
    return st.nonLayerDeviceBytes();
}

/// Device bytes this model cannot give up however small a budget it is handed:
/// what is left once every layer still on the device has been migrated to the
/// host. In practice the token embeddings, the LM head and the activation
/// scratch — none of which migrate.
///
/// The committed KV is deliberately NOT in here, and that is the whole point:
/// `migrateLayer` carries a layer's K/V to the host shadow along with its
/// weights, so offloading costs speed, not context. Reporting the committed
/// context as un-evictable (a formula over context length × every layer's dims)
/// keeps reserving the K/V of layers that are ALREADY on the host — VRAM nobody
/// is holding, subtracted from what a working image model is allowed to have.
/// Measured, so it falls as layers migrate and the next plan can hand over what
/// the last one freed.
///
/// A stepper with no split has nothing to migrate, so its floor is everything it
/// holds; `onDevice` reports true for every layer there, which gives exactly that
/// once the layer terms are subtracted.
pub fn evictionFloor(st: anytype) u64 {
    var evictable: u64 = 0;
    for (0..st.cfg.n_layers) |l| {
        if (!onDevice(st, l)) continue;
        evictable += st.layerWeightBytes(l) + layerKvBytes(st, l);
    }
    return st.be.deviceUsed() -| evictable;
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
/// shrinks per iteration as live slots drop, a fixed target can't express that.)
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
/// residency cap, through the SAME `vram.resolve` rule the GUI meter uses, so
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
/// ceiling, the enactment primitive the cross-model VRAM arbiter drives each
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
/// per-token OOM->offload->promote thrash cycle. Promoting late is harmless
/// (a settle retries next pump); promoting into the edge is not.
const promote_reserve: u64 = 256 << 20;

/// Free VRAM a promote must leave, for THIS model. The flat `promote_reserve`
/// above is only a floor: a forward pass allocates transient buffers whose size
/// depends on the architecture and batch (measured on gemma4-31B: a single
/// 1102 MiB request during prefill), and filling the ceiling to within 256 MiB of
/// the card guarantees the next one OOMs, which offloads layers, which is the
/// churn this path exists to avoid. The backend reports its own high-water, so
/// this is measured rather than guessed and needs no per-arch estimate.
///
/// Public because the VRAM arbiter must add the SAME figure on top of `demand`
/// when it sizes this model's ceiling (`vram.Participant.headroom`). Two
/// formulas would drift, and the direction they drift in is layers stranded on
/// the host: a ceiling that does not cover the reserve is a ceiling the loop
/// below cannot fill.
pub fn promoteReserve(st: anytype) u64 {
    return @max(promote_reserve, st.be.largestAllocation());
}

/// Migrate CPU layers back onto the GPU (LIFO by offload order), stopping before
/// the next would overflow `budget`, so the caller (VRAM coordinator, after
/// image generation) reclaims LLM residency while leaving room for whatever else
/// stays resident. Keeps the split armed (offload can fire again). Returns the
/// number promoted; 0 without a split.
///
/// Two DIFFERENT questions gate each promote, and conflating them is what used to
/// strand layers on the host:
///
///   does it fit under our CEILING?   accounting. Costs the layer's real bytes
///                                    (`foot`), nothing else, because that is all
///                                    the ceiling is counting.
///   does it fit on the CARD?         physical. Costs the layer's bytes plus
///                                    margin, so a promote that just barely fits
///                                    does not leave the next forward's transient
///                                    with nowhere to go.
///
/// Charging the margin to the ceiling too made the loop stop a whole margin's
/// worth of layers early, and (via `deferred`) a further `promote_slack` per
/// layer already promoted, against a ceiling the arbiter had sized to the model's
/// exact footprint. Nothing failed and nothing logged; the layers simply stayed on
/// the CPU. The card side is where a shortfall is real, and it is answerable: ask
/// the peer.
pub fn promoteBack(st: anytype, budget: u64) !usize {
    if (st.split == null) return 0;
    const sp = &st.split.?;
    var promoted: usize = 0;
    // A layer's WEIGHTS promote lazily (promoteLayer re-creates only its KV;
    // the weights re-cache pinned on the next forward), so deviceUsed() does
    // not yet include them. Carry that deferred cost across the loop,
    // without it one settle promoted a dozen layers, the next forward's
    // weight uploads OOM'd, the retry offloaded them all back, and the next
    // settle promoted again: a full per-token PCIe thrash cycle.
    var deferred: u64 = 0;
    while (sp.next > 0) {
        const l = sp.order[sp.next - 1];
        // `promoteCost` is the layer's bytes plus scheduling slack; the bytes
        // alone are what residency accounting (and so `need`/`demand`) counts.
        const foot = st.promoteCost(l) -| promote_slack;
        if (budget -| (st.be.deviceUsed() + deferred) < foot) break; // our ceiling binds
        const need_free = foot + promote_slack + promoteReserve(st);
        if (st.be.headroom() -| deferred < need_free) {
            // Room under our ceiling, but not on the CARD: another context is
            // holding it. On the GUI that is an idle image model whose residency
            // is pure cache, in a separate context our own eviction ladder cannot
            // reach. Ask for it.
            //
            // This rung existed only under `tensorCreate`'s OOM handler, which
            // cannot fire here: declining to promote allocates nothing, so there
            // is no failure to react to. The model just stayed on the CPU beside
            // an idle peer that would have yielded for the asking.
            _ = st.be.requestForeignRoom(need_free -| (st.be.headroom() -| deferred));
            if (st.be.headroom() -| deferred < need_free) break; // the peer had nothing
        }
        const before = st.be.deviceUsed();
        try st.promoteLayer(l);
        // Whatever promoteLayer did NOT allocate now (its weight bytes) arrives at
        // the next forward; count it as spent already. Slack-free, so the ceiling
        // check above does not drift further out with every layer promoted.
        deferred += foot -| (st.be.deviceUsed() -| before);
        sp.next -= 1;
        promoted += 1;
    }
    return promoted;
}

// --- tests -----------------------------------------------------------------

/// A stepper stand-in for the promote scheduler: enough of the duck-typed
/// contract above to drive `promoteBack`, with a card model behind it. Layer
/// sizes are a 1-bit 27B's (~60 MiB of weights, a few MiB of K/V), the shape
/// where the reserve arithmetic bites hardest: a 256 MiB margin charged to the
/// ceiling is four layers, and a GiB-scale one is most of the model.
const MockLm = struct {
    const weight_bytes: u64 = 60 << 20;
    const kv_bytes: u64 = 4 << 20;

    cfg: struct { n_layers: usize },
    split: ?Split,
    be: MockBe,

    const Split = struct {
        dynamic: bool = true,
        budget: u64 = 0,
        order: []usize,
        next: usize,
        n_cpu: usize,
    };

    const MockBe = struct {
        /// Bytes WE hold on the card.
        used: u64 = 0,
        card: u64 = 64 << 30,
        /// Bytes another device context holds, yieldable unless `peer_locked`.
        peer: u64 = 0,
        peer_locked: bool = false,
        /// Largest single allocation seen; drives `promoteReserve`.
        largest: u64 = 0,
        /// Cached model parameters inside `used`. The loader uploads weights
        /// before the split is planned, so this is the term `splitHostCount`
        /// must not charge to the budget twice.
        pinned: u64 = 0,

        fn deviceUsed(self: *MockBe) u64 {
            return self.used;
        }
        fn pinnedWeightBytes(self: *MockBe) u64 {
            return self.pinned;
        }
        fn evictableWeightBytes(_: *MockBe) u64 {
            return 0; // LLM weights never stream; they pin or they offload
        }
        /// No 0.9 fudge, unlike the real backend: the arithmetic under test is
        /// about which term binds, and a fudge factor only obscures it.
        fn headroom(self: *MockBe) u64 {
            return self.card -| self.used -| self.peer;
        }
        fn largestAllocation(self: *MockBe) u64 {
            return self.largest;
        }
        fn requestForeignRoom(self: *MockBe, needed: u64) u64 {
            if (self.peer_locked) return 0;
            const got = @min(needed, self.peer);
            self.peer -= got;
            return got;
        }
    };

    fn layerWeightBytes(_: *MockLm, _: usize) u64 {
        return weight_bytes;
    }
    fn promoteCost(_: *MockLm, _: usize) usize {
        return weight_bytes + kv_bytes + promote_slack;
    }
    /// Mirrors the real steppers: only the K/V is allocated now, the weights
    /// re-cache on the next forward (which is what `deferred` carries).
    fn promoteLayer(self: *MockLm, _: usize) !void {
        self.be.used += kv_bytes;
        self.split.?.n_cpu -= 1;
    }

    fn init(order: []usize, be: MockBe) MockLm {
        return .{
            .cfg = .{ .n_layers = order.len },
            .split = .{ .order = order, .next = order.len, .n_cpu = order.len },
            .be = be,
        };
    }
};

/// `n` layers, all on the host, in descending offload order.
fn mockOrder(buf: []usize) []usize {
    for (buf, 0..) |*o, i| o.* = buf.len - 1 - i;
    return buf;
}

test "residency.promoteBack: a ceiling at the model's own footprint fills completely" {
    // The stranded-layer bug, at its source. `promoteCost` carries scheduling
    // slack and `promoteReserve` adds a margin, both of which are about PHYSICAL
    // room; charging them to the accounting ceiling as well stopped the loop with
    // most of the model still on the host, under a ceiling the arbiter had sized
    // to exactly the model's footprint. `deferred` then compounded it, adding
    // another slack per layer already promoted.
    var buf: [8]usize = undefined;
    var st = MockLm.init(mockOrder(&buf), .{}); // empty card, nothing else resident
    const foot = MockLm.weight_bytes + MockLm.kv_bytes;
    // Exactly what `residency.need` would report for these 8 layers: no slack, no
    // reserve. This is what the arbiter publishes.
    try std.testing.expectEqual(@as(usize, 8), try promoteBack(&st, 8 * foot));
    try std.testing.expectEqual(@as(usize, 0), st.split.?.next);
    try std.testing.expectEqual(@as(usize, 0), st.split.?.n_cpu);
}

test "residency.promoteBack: our own ceiling still binds" {
    // The counterpart. A ceiling below the footprint must still stop the loop,
    // or the arbiter's split handle would mean nothing.
    var buf: [8]usize = undefined;
    var st = MockLm.init(mockOrder(&buf), .{});
    const foot = MockLm.weight_bytes + MockLm.kv_bytes;
    try std.testing.expectEqual(@as(usize, 3), try promoteBack(&st, 3 * foot));
    try std.testing.expectEqual(@as(usize, 5), st.split.?.n_cpu);
}

test "residency.promoteBack: a card held by another context is asked, not surrendered to" {
    // The reported failure. The ceiling has room, the CARD does not, and what is
    // holding it is an idle image model in a separate device context. Declining
    // allocates nothing, so `tensorCreate`'s OOM ladder never fires and the peer
    // is never asked; the layers just stay on the CPU. The peer has to be asked
    // from here.
    var buf: [8]usize = undefined;
    const foot = MockLm.weight_bytes + MockLm.kv_bytes;
    var st = MockLm.init(mockOrder(&buf), .{
        .card = 4 << 30,
        .peer = (4 << 30) - (128 << 20), // 128 MiB free: under the 256 MiB reserve
    });
    try std.testing.expectEqual(@as(usize, 8), try promoteBack(&st, 8 * foot));
    try std.testing.expect(st.be.peer < (4 << 30) - (128 << 20)); // it yielded

    // A peer that cannot yield (busy, or its residency mutex is held) stops the
    // loop rather than promoting into a full card.
    var buf2: [8]usize = undefined;
    var st2 = MockLm.init(mockOrder(&buf2), .{
        .card = 4 << 30,
        .peer = (4 << 30) - (128 << 20),
        .peer_locked = true,
    });
    try std.testing.expectEqual(@as(usize, 0), try promoteBack(&st2, 8 * foot));
    try std.testing.expectEqual(@as(u64, (4 << 30) - (128 << 20)), st2.be.peer);
}

test "residency.promoteBack: a big transient raises the physical margin, not the ceiling cost" {
    // `largestAllocation` is a monotonic session high-water, and one image prefill
    // pushes it into the GiB range for good. That must show up as free VRAM the
    // promote leaves on the card, never as ceiling the layers are charged for.
    var buf: [8]usize = undefined;
    const foot = MockLm.weight_bytes + MockLm.kv_bytes;
    var st = MockLm.init(mockOrder(&buf), .{ .largest = 1 << 30 });
    try std.testing.expectEqual(@as(usize, 8), try promoteBack(&st, 8 * foot));

    // ...but a card that cannot spare that margin does stop it.
    var buf2: [8]usize = undefined;
    var st2 = MockLm.init(mockOrder(&buf2), .{ .card = 8 * foot + (512 << 20), .largest = 1 << 30 });
    try std.testing.expectEqual(@as(usize, 0), try promoteBack(&st2, 8 * foot));
}

/// A 60-layer model whose weights the loader has already uploaded, i.e. the
/// state `autoOffload` actually plans from. Layer bytes and the head are a
/// gemma4-31B's, the shape the reported failure was on.
const SplitCase = struct {
    const layer: u64 = 276 << 20; // 60 layers ~ 16.2 GiB of weights
    const head: u64 = 700 << 20;
    const other: u64 = 1500 << 20; // committed KV + activation scratch
    const n = 60;

    per: [n]usize = @splat(layer),
    order: [n]usize = undefined,
    st: MockLm = undefined,

    /// Ready to plan on a `card`-byte GPU on which the loader has already
    /// uploaded the head and `uploaded` of the layers.
    /// `promoteReserve`'s floor, which every dynamic plan now leaves free.
    const transient: u64 = 256 << 20;

    fn init(self: *SplitCase, card: u64, uploaded: u64) void {
        self.per = @splat(layer);
        for (&self.order, 0..) |*o, i| o.* = n - 1 - i;
        const weights = uploaded * layer + head;
        self.st = .{
            .cfg = .{ .n_layers = n },
            .split = null,
            .be = .{ .used = weights + other, .card = card, .pinned = weights },
        };
    }
    fn plan(self: *SplitCase, budget: u64) usize {
        return splitHostCount(&self.st, budget, head, true, &self.per, &self.order);
    }
};

test "residency.evictionFloor: only the bytes that cannot migrate" {
    // The reported symptom, from the other side: an image model squeezed by an
    // idle LLM. The floor is what the arbiter refuses to take, so anything in it
    // that CAN be given up is VRAM a working image model never gets offered.
    var buf: [8]usize = undefined;
    var st = MockLm.init(mockOrder(&buf), .{});
    const foot = MockLm.weight_bytes + MockLm.kv_bytes;
    const residue: u64 = 900 << 20; // embeddings + LM head + scratch
    // Fully resident: 8 layers on the card plus the residue.
    st.split.?.next = 0;
    st.split.?.n_cpu = 0;
    st.be.used = 8 * foot + residue;
    try std.testing.expectEqual(residue, evictionFloor(&st));

    // Five layers migrate to the host: their weights AND their K/V go with them,
    // so the floor must not follow the conversation, only the residue stays.
    st.split.?.next = 5;
    st.split.?.n_cpu = 5;
    st.be.used = 3 * foot + residue;
    try std.testing.expectEqual(residue, evictionFloor(&st));

    // Everything on the host: nothing left to give.
    st.split.?.next = 8;
    st.split.?.n_cpu = 8;
    st.be.used = residue;
    try std.testing.expectEqual(residue, evictionFloor(&st));
}

test "residency.evictionFloor: no split means nothing migrates" {
    // A stepper with no split has no host shadow to move layers into, so
    // everything it holds is its floor. (`onDevice` reports true for every layer
    // there, and the layer terms then cancel what it is holding.)
    var st = MockLm.init(&.{}, .{});
    st.split = null;
    st.cfg.n_layers = 0;
    st.be.used = 4 << 30;
    try std.testing.expectEqual(@as(u64, 4 << 30), evictionFloor(&st));
}

test "residency.splitHostCount: weights already on the card are not charged twice" {
    // The reported failure: a 31B loads, its weights upload, and the split
    // planner then reads `deviceUsed()` ~18 GiB and concludes there is no room
    // for the very weights that number is made of. 58 of 60 layers went to the
    // host under a budget that fit the whole model, and because a host layer's
    // weights are reclaimed lazily they stayed in VRAM anyway: prefill ran on the
    // CPU at 15 tok/s and freed nothing.
    var c: SplitCase = undefined;
    c.init(24101 << 20, SplitCase.n); // every layer uploaded, as pread loading leaves it
    // The GUI's ceiling for this card, comfortably above the model's footprint.
    try std.testing.expectEqual(@as(usize, 0), c.plan(21848 << 20));
}

test "residency.splitHostCount: a budget below the model still places layers" {
    // The counterpart: the accounting fix must not turn the planner off. A
    // budget with room for 40 layers' weights offloads the other 20.
    var c: SplitCase = undefined;
    c.init(24101 << 20, SplitCase.n);
    try std.testing.expectEqual(@as(usize, 20), c.plan(SplitCase.other + SplitCase.head + SplitCase.transient + 40 * SplitCase.layer));
}

test "residency.splitHostCount: the card's live free VRAM binds under a generous budget" {
    // `budget` is an abstract ceiling; the card is physical. A model too big for
    // the card gets only the layers that fit uploaded, and an unlimited budget
    // must still plan the rest onto the host rather than trusting the number.
    var c: SplitCase = undefined;
    c.init(SplitCase.other + SplitCase.head + SplitCase.transient + 30 * SplitCase.layer, 30);
    try std.testing.expectEqual(@as(usize, 30), c.plan(std.math.maxInt(u64)));
}

test "residency.splitHostCount: a dynamic plan leaves the forward's transient free" {
    // The second half of the reported failure. With the weight accounting fixed
    // the planner packed a 31B to within ~300 MiB of the ceiling, and the first
    // forward's 220 MiB buffer had nowhere to go: `tensorCreate` failed, the OOM
    // handler dumped 4 layers to the host, and prefill ran at 115 tok/s. Leaving
    // `promoteReserve` free costs a layer, up front, where the planner picks it.
    var c: SplitCase = undefined;
    c.init(24101 << 20, SplitCase.n);
    // A budget with room for every layer, the head and `other` -- and nothing
    // over. Exactly the shape that OOM'd.
    const snug = SplitCase.other + SplitCase.head + SplitCase.n * SplitCase.layer;
    try std.testing.expectEqual(@as(usize, 1), c.plan(snug));

    // A static split is left alone at the same budget: it reserves the KV at
    // capacity plus its own slack instead, so charging the transient on top
    // would offload twice for the same bytes.
    try std.testing.expectEqual(@as(usize, 0), splitHostCount(&c.st, snug, SplitCase.head, false, &c.per, &c.order));
}

test "residency.hostWeightBytes counts the layers actually migrated, not the plan" {
    var buf: [8]usize = undefined;
    var st = MockLm.init(mockOrder(&buf), .{});

    // Every layer on the host.
    try std.testing.expectEqual(8 * MockLm.weight_bytes, hostWeightBytes(&st));

    // A dynamic split part-way through: `next` is how far migration has actually
    // got, and that -- not the planned `n_cpu` -- is what the bar must report, or
    // it claims weights are on the CPU before they have moved.
    st.split.?.next = 3;
    try std.testing.expectEqual(3 * MockLm.weight_bytes, hostWeightBytes(&st));

    st.split.?.next = 0;
    try std.testing.expectEqual(@as(u64, 0), hostWeightBytes(&st));

    // No split armed at all: fully resident, and no per-layer walk to get it wrong.
    st.split = null;
    try std.testing.expectEqual(@as(u64, 0), hostWeightBytes(&st));
}

test "residency.hostWeightBytes is the weights only, not the K/V they carry" {
    var buf: [4]usize = undefined;
    var st = MockLm.init(mockOrder(&buf), .{});
    // `measured`/`need` fold K/V in; this figure deliberately does not, so it
    // holds still while a conversation grows.
    const host = hostWeightBytes(&st);
    errdefer std.debug.print("host={d} weights={d} kv={d}\n", .{ host, MockLm.weight_bytes, MockLm.kv_bytes });
    try std.testing.expectEqual(4 * MockLm.weight_bytes, host);
    try std.testing.expect(host < 4 * (MockLm.weight_bytes + MockLm.kv_bytes));
}
