//! Cross-model VRAM coordination.
//!
//! The GUI runs an LLM and a diffusion model on the same device and must
//! arbitrate VRAM between them (a big image push must be able to migrate LLM
//! layers to the host, and reclaim them when it finishes). The pieces here are
//! backend/model-agnostic and live in the library so both frontends can use
//! them (the diffusion CLI and tp-llm have a single model and need no arbiter,
//! but the primitives are shared).
//!
//! Design constraint that shapes everything: **a CUDA context is bound
//! per-thread** — each model's device residency may only be mutated on the
//! worker thread that bound its context. So the arbiter never touches the GPU
//! directly. It computes a desired residency budget per model and publishes it
//! to that model's `ControlPoint`; the model's own worker observes the intent
//! at a safe boundary (between LLM tokens / diffusion steps) and applies it on
//! its own thread. That eliminates cross-thread device races by construction —
//! the failure mode of the ad-hoc `imageVramEnter`/`settleLlm` hooks this
//! replaces.

const std = @import("std");

/// A cooperative control point between a model's compute WORKER thread and an
/// external COORDINATOR (the app-level `VramArbiter`; later also a pause UI).
/// One per participant, embedded in the model's session. The coordinator
/// publishes *intents* from any thread; the worker *polls* them at its own safe
/// boundaries and acts on its own thread. Nothing here touches the device.
///
/// Two intents:
///   - `budget` — the desired device-residency ceiling (bytes) the worker
///     should settle toward. Persistent + last-write-wins (it is also the
///     ongoing growth ceiling, not a one-shot), `unconstrained` = no limit.
///   - `pause` — reserved for the upcoming pause feature (park the worker at
///     its next safe boundary). Published + observable now; not yet consumed by
///     any worker, so it is inert until the pause handshake lands. Wiring it in
///     from day one keeps that a fill-in rather than a re-architecture.
pub const ControlPoint = struct {
    budget: std.atomic.Value(u64) = .init(unconstrained),
    pause: std.atomic.Value(bool) = .init(false),

    /// `budget` sentinel: no residency limit (use as much VRAM as available).
    pub const unconstrained: u64 = std.math.maxInt(u64);

    // --- coordinator side (any thread) ---

    /// Ask the worker to settle to `bytes` of device residency at its next safe
    /// boundary. Pass `unconstrained` (or `clearBudget`) to lift the limit.
    pub fn requestBudget(self: *ControlPoint, bytes: u64) void {
        self.budget.store(bytes, .release);
    }

    pub fn clearBudget(self: *ControlPoint) void {
        self.budget.store(unconstrained, .release);
    }

    /// Reserved: request/lift a pause. Inert until a worker consumes it.
    pub fn requestPause(self: *ControlPoint, on: bool) void {
        self.pause.store(on, .release);
    }

    // --- worker side (the model's own compute thread) ---

    /// The pending residency ceiling (bytes), or null when unconstrained. The
    /// worker calls this at a safe boundary and settles its device residency
    /// toward the returned target. Persistent (a peek, not a take): the target
    /// stays in effect — settle ops are idempotent, so re-observing a satisfied
    /// target is a cheap no-op.
    pub fn budgetTarget(self: *const ControlPoint) ?u64 {
        const b = self.budget.load(.acquire);
        return if (b == unconstrained) null else b;
    }

    /// Reserved: whether a pause has been requested (for the future handshake).
    pub fn pausePending(self: *const ControlPoint) bool {
        return self.pause.load(.acquire);
    }
};

/// A model the arbiter can drive, behind a stable vtable so LLM and diffusion
/// look identical to the arbiter. Read methods (`usage`/`demand`/`floor`/`busy`)
/// may be called from any thread; `applyBudget` mutates device residency and so
/// is only invoked on a thread that may bind this model's context — the idle path
/// (arbiter thread) or the model's own worker (via `pollAndApply`).
pub const Participant = struct {
    ctx: *anyopaque,
    control: *ControlPoint,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Device bytes this model currently holds.
        usage: *const fn (ctx: *anyopaque) u64,
        /// Device bytes this model would hold if NOTHING contended: its full
        /// resident footprint (LLM: every layer back on the GPU + KV at
        /// capacity; diffusion: the whole pipeline resident). Always `>= usage`.
        ///
        /// This exists because `usage` alone cannot tell "8 GiB is all I want"
        /// apart from "8 GiB is all I was allowed", and the whole
        /// idle-diffusion-yields-to-the-LLM case turns on that distinction. The
        /// old policy computed each model's target from the OTHER model's
        /// *usage*, which made the pair algebraically cancel: the LLM was
        /// settled to `limit − diff_usage`, so diffusion was then offered
        /// `limit − llm_usage == diff_usage` — its own residency back, i.e. a
        /// guaranteed no-op, every time. See `Arbiter.plan`.
        demand: *const fn (ctx: *anyopaque) u64,
        /// Bytes that cannot be evicted (LLM: committed KV; diffusion: its whole
        /// working set while an image is in flight, since mid-image eviction
        /// would force per-step streaming). The arbiter never targets a model
        /// below its floor.
        floor: *const fn (ctx: *anyopaque) u64,
        /// Is a compute worker running on this model's context right now?
        busy: *const fn (ctx: *anyopaque) bool,
        /// Settle device residency to `target` bytes and set that as the ongoing
        /// growth ceiling. Idempotent (a satisfied target is a no-op). Caller
        /// guarantees it runs on a thread that may bind this model's context.
        applyBudget: *const fn (ctx: *anyopaque, target: u64) void,
    };

    pub fn usage(self: Participant) u64 {
        return self.vtable.usage(self.ctx);
    }
    pub fn demand(self: Participant) u64 {
        return @max(self.vtable.demand(self.ctx), self.vtable.usage(self.ctx));
    }
    pub fn floor(self: Participant) u64 {
        return self.vtable.floor(self.ctx);
    }
    pub fn busy(self: Participant) bool {
        return self.vtable.busy(self.ctx);
    }

    /// Drive this model toward `target` bytes (clamped up to its floor). This is
    /// the ONE place the idle/busy split lives: the desired ceiling is always
    /// published to the control point (source of truth for both the growth path
    /// and a worker that starts later); if no worker is running we also apply it
    /// now on the caller's thread, otherwise the worker applies it at its next
    /// safe boundary via `pollAndApply`.
    ///
    /// A `target` of 0 is IGNORED (nothing published or applied): 0 means the
    /// arbiter has no real budget (uninitialized, or a fully-collapsed
    /// measurement), and clamping it up to the floor would manufacture a real —
    /// and typically unreachable — ceiling out of garbage, evicting the whole
    /// model (the qwen3-32B first-message mass-offload bug). A coordinator that
    /// wants a model to yield everything it can targets a small nonzero budget;
    /// the floor clamp keeps that safe.
    pub fn settle(self: Participant, target: u64) void {
        if (target == 0) {
            std.log.debug("[vram] settle skipped: zero target (would clamp to floor {d} MiB)", .{self.floor() >> 20});
            return;
        }
        const clamped = @max(target, self.floor());
        if (clamped != target)
            std.log.debug("[vram] settle target {d} MiB below the committed-KV floor; raised to {d} MiB", .{ target >> 20, clamped >> 20 });
        self.control.requestBudget(clamped);
        if (!self.busy()) self.vtable.applyBudget(self.ctx, clamped);
    }

    /// Worker-side: at a safe boundary, enact any pending published ceiling on
    /// this (the model's own) thread. Cheap no-op when already settled.
    pub fn pollAndApply(self: Participant) void {
        if (self.control.budgetTarget()) |t| self.vtable.applyBudget(self.ctx, t);
    }
};

/// Which model a cross-model request is on behalf of.
pub const Side = enum { llm, diffusion };

/// A coherent residency target for BOTH models, computed in one pass.
pub const Plan = struct {
    llm: u64,
    diffusion: u64,
};

/// The single owner of "how much VRAM each model may hold." Replaces the ad-hoc
/// enter/exit/budget/reclaim hooks whose three code paths settled the LLM to
/// three disagreeing targets and no-op'd whenever it was busy. Every relevant
/// event (image-queue edge, generation start/stop, meter commit) calls one of
/// the mutators, which recomputes the authoritative targets and drives them
/// through `Participant.settle`.
///
/// Both models are full `Participant`s: `plan` computes their targets TOGETHER
/// and `rebalance` enacts them shrink-before-grow. This is the fix for the
/// structural defect behind the "diffusion never yields to the LLM" report —
/// the previous version drove only the LLM (to `limit − diffusion's current
/// usage`, i.e. treating an idle image model's opportunistic cache as a hard
/// reservation) and left the reverse direction to a separate, later call in the
/// app that computed its target from the LLM's *just-shrunk* usage and therefore
/// always cancelled to a no-op. Two targets derived from each other's live usage
/// are not a plan; see `plan`.
pub const Arbiter = struct {
    llm: ?Participant = null,
    diffusion: ?Participant = null,
    /// Total VRAM ceiling the models share (the meter's `limit` handle × card).
    limit: u64 = 0,
    /// The LLM's guaranteed floor under contention (the meter's `split` handle).
    llm_share: u64 = 0,
    /// Diffusion has queued/running work that wants VRAM (queue non-empty).
    /// Genuinely external state — the queue is not derivable from residency —
    /// unlike the old `diff_used` mirror of `diffusion.usage()`, which was
    /// hand-updated from three call sites and went stale between them.
    diff_active: bool = false,

    /// Compute both models' residency targets from one snapshot of (demand,
    /// floor) plus the meter's ceiling/split. Pure: reads participants, mutates
    /// nothing. Returns null when there is no real ceiling to plan against
    /// (`limit == 0`: `setBudgets` never ran, or the VRAM query behind it
    /// failed) — driving residency from that would publish garbage, which is how
    /// the qwen3-32B first-message mass-offload happened.
    ///
    /// Policy, in priority order:
    ///
    ///  1. **No contention** (both demands fit under the ceiling) — everyone
    ///     stays fully resident. Covers the single-model case too: an absent
    ///     participant contributes 0 demand and 0 floor.
    ///  2. **Diffusion active** (something queued or generating) — diffusion
    ///     gets the room, except that the meter's split handle guarantees the
    ///     LLM `llm_share` no matter what. The LLM yields even mid-generation:
    ///     the target is published to its control point and applied at its next
    ///     token boundary.
    ///  3. **Diffusion idle** — its residency is *opportunistic cache* (the next
    ///     image re-uploads what it needs), so the LLM has priority and
    ///     diffusion yields exactly the deficit: `limit − llm_demand`, never
    ///     below its own floor, never above what it wants. This is the direction
    ///     that never worked.
    ///
    /// Each side is clamped into `[floor, demand]`, so a target can neither evict
    /// something un-evictable nor inflate a model past its footprint.
    pub fn plan(self: *const Arbiter) ?Plan {
        if (self.limit == 0) return null;

        const l_dem = if (self.llm) |p| p.demand() else 0;
        const l_flr = if (self.llm) |p| p.floor() else 0;
        const d_flr = if (self.diffusion) |p| p.floor() else 0;
        // An ACTIVE queue is entitled to the complement of the LLM's guaranteed
        // share whether or not we can size the image model yet — this is the whole
        // meaning of the split handle. Floored rather than measured because at the
        // queue-start edge the pipeline is not loaded (`usage` is 0) and the
        // estimate behind `demand` reads checkpoint file sizes, which can fail; a
        // demand of 0 there would read as "no contention" and leave the LLM holding
        // the entire card for the image model to load into.
        const d_dem = blk: {
            const raw = if (self.diffusion) |p| p.demand() else 0;
            break :blk if (self.diff_active) @max(raw, self.limit -| self.llm_share) else raw;
        };

        var out: Plan = if (l_dem + d_dem <= self.limit)
            .{ .llm = l_dem, .diffusion = d_dem } // (1) uncontended
        else if (self.diff_active) blk: { // (2) diffusion has work
            const l = clamp(self.llm_share, l_flr, l_dem);
            break :blk .{ .llm = l, .diffusion = clamp(self.limit -| l, d_flr, d_dem) };
        } else blk: { // (3) diffusion is idle cache
            const d = clamp(self.limit -| l_dem, d_flr, d_dem);
            break :blk .{ .llm = clamp(self.limit -| d, l_flr, l_dem), .diffusion = d };
        };
        // "Hold nothing" is a legitimate target for DIFFUSION — its residency is
        // pure cache — but 0 is also the sentinel the enactment paths read as "no
        // target, do nothing". Ask for 1 byte instead; the incremental evict then
        // frees `usage − 1`, i.e. everything.
        //
        // The LLM deliberately gets NO such floor. A 0 there means the plan had
        // nothing to give it (the ceiling dragged below what a mid-image diffusion
        // holds, or the split handle at zero with no committed KV yet to floor it),
        // and turning that into a real 1-byte ceiling would offload every layer to
        // the host chasing it — the qwen3-32B mass-offload bug. `settle` rejects a
        // zero target, so the LLM simply keeps what it has and recovers through the
        // reactive `requestRoom` path instead.
        out.diffusion = @max(out.diffusion, 1);
        return out;
    }

    /// Recompute + drive both models' authoritative residency targets.
    /// Idempotent; safe to call on any event.
    ///
    /// The two targets come from ONE plan, but they must be ENACTED in order:
    /// the model that grows would otherwise allocate into room the other has not
    /// released yet — i.e. into a full card. Shrink first, grow second. Ordering
    /// on "is this target below what the model currently holds" gets that right
    /// in every combination and is stable when neither side shrinks.
    pub fn rebalance(self: *Arbiter) void {
        const p = self.plan() orelse {
            // Only anomalous when there IS an LLM: the meter policy resolves against
            // a live session, so `limit == 0` with no LLM is just image-studio mode
            // (diffusion gets the AUTO budget) and must not warn on every image.
            if (self.llm != null)
                std.log.warn("[vram] rebalance skipped: budgets uninitialized (meter policy never resolved — VRAM query failed?)", .{});
            return;
        };
        const llm_shrinks = if (self.llm) |q| p.llm < q.usage() else false;
        if (llm_shrinks) {
            if (self.llm) |q| q.settle(p.llm);
            if (self.diffusion) |q| q.settle(p.diffusion);
        } else {
            if (self.diffusion) |q| q.settle(p.diffusion);
            if (self.llm) |q| q.settle(p.llm);
        }
    }

    /// Diffusion queue started (`true`) / drained (`false`). Triggers a rebalance
    /// so the LLM starts yielding immediately, before the image model loads.
    pub fn setDiffusionActive(self: *Arbiter, active: bool) void {
        self.diff_active = active;
        self.rebalance();
    }

    /// Update the resolved ceiling + LLM floor (meter drag commit) and rebalance.
    pub fn setBudgets(self: *Arbiter, limit: u64, llm_share: u64) void {
        self.limit = limit;
        self.llm_share = llm_share;
        self.rebalance();
    }

    /// REACTIVE cross-model reclaim: `side` is about to allocate and is `bytes`
    /// short of what its own context can give it. Ask the OTHER model to free
    /// that much on top of what it already released, and report the bytes
    /// actually freed (0 = nothing available).
    ///
    /// This is the missing mirror of the diffusion pipeline's `Reclaim` hook,
    /// which has always been able to migrate LLM layers to the host mid-image.
    /// In the other direction there was nothing: because the two models hold
    /// their VRAM in SEPARATE device contexts, the LLM backend's own eviction
    /// ladder cannot reach diffusion's weights, and LLM weights are all pinned
    /// so that ladder has nothing of its own to drop either. A resident-but-idle
    /// image model was therefore an immovable reservation the LLM could only
    /// work around by offloading its own layers to the CPU — or, when the
    /// allocation wasn't a layer it could migrate, by failing the turn outright
    /// with DeviceOutOfMemory.
    ///
    /// Only an IDLE peer is touched: the apply binds the peer's device context,
    /// which its own worker would otherwise be using. Callable from either
    /// model's worker thread, so it deliberately does NOT publish a new ceiling
    /// to the peer's control point — `rebalance` stays the single owner of the
    /// steady-state targets, and this is purely "free something now".
    pub fn requestRoom(self: *Arbiter, side: Side, bytes: u64) u64 {
        if (bytes == 0) return 0;
        const peer = (switch (side) {
            .llm => self.diffusion,
            .diffusion => self.llm,
        }) orelse return 0;
        if (peer.busy()) {
            std.log.debug("[vram] {t} asked for {d} MiB: peer is busy, declined", .{ side, bytes >> 20 });
            return 0;
        }
        const used = peer.usage();
        const target = @max(used -| bytes, @max(peer.floor(), 1));
        if (target >= used) return 0;
        peer.vtable.applyBudget(peer.ctx, target);
        const freed = used -| peer.usage();
        std.log.info("[vram] {t} asked for {d} MiB: peer yielded {d} MiB ({d}→{d} MiB resident)", .{
            side, bytes >> 20, freed >> 20, used >> 20, peer.usage() >> 20,
        });
        return freed;
    }

    /// The resident-weight budget the next image may pin: the room the plan
    /// leaves once the LLM is at its planned target, so it agrees with what
    /// `rebalance` is driving the LLM toward (the two used to be separate
    /// formulas that disagreed, one reading the LLM's live usage and the other
    /// its share). Deliberately the *allowance* rather than `plan.diffusion` —
    /// diffusion's own `demand` is an estimate from file sizes, and an estimate
    /// that undershoots the real resident size would pin part of the image model
    /// and stream the rest. Floored so a tiny allowance still streams rather than
    /// reading as the AUTO sentinel.
    ///
    /// With no plan (`limit == 0`: `setBudgets` never ran because there is no LLM
    /// session for the meter policy to resolve against, i.e. pure image-studio
    /// mode) return 0 = the pipeline's AUTO sentinel (pin what fits live free
    /// VRAM). Returning the 256 MiB floor here instead pinned a sliver of the
    /// image model and evicted/streamed the rest.
    pub fn diffusionBudget(self: *const Arbiter) u64 {
        const p = self.plan() orelse {
            std.log.debug("[vram] diffusion budget: arbiter uninitialized (no LLM/meter policy) → auto (pin what fits)", .{});
            return 0;
        };
        const min_budget: u64 = 256 << 20;
        return @max(min_budget, self.limit -| p.llm);
    }
};

fn clamp(v: u64, lo: u64, hi: u64) u64 {
    return @min(@max(v, lo), @max(lo, hi));
}

// --- tests -----------------------------------------------------------------

/// A mock model standing in for an LLM/diffusion session, recording the last
/// `applyBudget` it was driven to and whether it was applied directly.
/// `applyBudget` also shrinks `used` toward the target (never below `floor_b`),
/// like both real participants do, so ordering-sensitive behaviour is testable.
const MockModel = struct {
    used: u64,
    /// Footprint if unconstrained. 0 = "whatever I currently hold" (the common
    /// fully-resident case); `Participant.demand` maxes it with `used`.
    want: u64 = 0,
    floor_b: u64 = 0,
    is_busy: bool = false,
    applied: ?u64 = null, // last applyBudget target (null = never applied directly)
    /// applyBudget targets in call order across ALL mocks, for ordering asserts.
    var_log: *std.ArrayList(Entry) = undefined,
    has_log: bool = false,
    name: []const u8 = "?",
    cp: ControlPoint = .{},

    const Entry = struct { name: []const u8, target: u64 };

    fn usageFn(ctx: *anyopaque) u64 {
        return mock(ctx).used;
    }
    fn demandFn(ctx: *anyopaque) u64 {
        return mock(ctx).want;
    }
    fn floorFn(ctx: *anyopaque) u64 {
        return mock(ctx).floor_b;
    }
    fn busyFn(ctx: *anyopaque) bool {
        return mock(ctx).is_busy;
    }
    fn applyFn(ctx: *anyopaque, target: u64) void {
        const m = mock(ctx);
        m.applied = target;
        if (m.has_log) m.var_log.append(std.testing.allocator, .{ .name = m.name, .target = target }) catch {};
        // Real participants only ever move toward the target: shrink to it, or
        // grow up to their demand. Mirror that so `used` stays believable.
        if (target < m.used) m.used = @max(target, m.floor_b) else m.used = @min(target, @max(m.want, m.used));
    }
    fn mock(ctx: *anyopaque) *MockModel {
        return @ptrCast(@alignCast(ctx));
    }
    const vtable: Participant.VTable = .{
        .usage = usageFn,
        .demand = demandFn,
        .floor = floorFn,
        .busy = busyFn,
        .applyBudget = applyFn,
    };
    fn participant(self: *MockModel) Participant {
        return .{ .ctx = self, .control = &self.cp, .vtable = &vtable };
    }
};

test "Participant.settle: idle applies now, busy defers to control point" {
    var m: MockModel = .{ .used = 8 << 30, .floor_b = 1 << 30 };

    // Idle: applied directly AND published.
    m.participant().settle(3 << 30);
    try std.testing.expectEqual(@as(?u64, 3 << 30), m.applied);
    try std.testing.expectEqual(@as(?u64, 3 << 30), m.cp.budgetTarget());

    // Busy: published only; NOT applied directly (would race the worker's context).
    m.applied = null;
    m.is_busy = true;
    m.participant().settle(2 << 30);
    try std.testing.expectEqual(@as(?u64, null), m.applied);
    try std.testing.expectEqual(@as(?u64, 2 << 30), m.cp.budgetTarget());

    // The busy worker later enacts it on its own thread.
    m.participant().pollAndApply();
    try std.testing.expectEqual(@as(?u64, 2 << 30), m.applied);
}

test "Participant.settle: target is clamped up to the floor" {
    var m: MockModel = .{ .used = 8 << 30, .floor_b = 4 << 30 };
    m.participant().settle(1 << 30); // below floor
    try std.testing.expectEqual(@as(?u64, 4 << 30), m.applied);
    try std.testing.expectEqual(@as(?u64, 4 << 30), m.cp.budgetTarget());
}

test "Arbiter: diffusion active drives the LLM down to its share; idle frees it" {
    // The LLM wants the whole ceiling; no diffusion participant registered.
    var m: MockModel = .{ .used = 20 << 30, .want = 22 << 30, .floor_b = 2 << 30 };
    var arb: Arbiter = .{ .llm = m.participant(), .limit = 22 << 30, .llm_share = 6 << 30 };

    // Diffusion starts → LLM yields to its share immediately.
    arb.setDiffusionActive(true);
    try std.testing.expectEqual(@as(?u64, 6 << 30), m.applied);
    // Diffusion may plan for limit − share even though the LLM only just started
    // coming down (the VAE reclaim ladder covers the transient).
    try std.testing.expectEqual(@as(u64, 16 << 30), arb.diffusionBudget());

    // Diffusion drains → LLM may reclaim up to the whole limit.
    arb.setDiffusionActive(false);
    try std.testing.expectEqual(@as(?u64, 22 << 30), m.applied);
}

test "Arbiter: an IDLE resident image model yields to the LLM (TODO #1)" {
    // The reported bug. Diffusion holds 8 GiB of a 22 GiB budget and is idle; the
    // LLM wants 18 GiB. Old behaviour: the LLM was settled to limit − 8 = 14 GiB
    // (offloading layers to the CPU) and diffusion was then offered
    // limit − 14 = 8 GiB — exactly what it already held, so it never freed a byte.
    var llm: MockModel = .{ .used = 14 << 30, .want = 18 << 30, .floor_b = 2 << 30, .name = "llm" };
    var diff: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .floor_b = 0, .name = "diff" };
    var log: std.ArrayList(MockModel.Entry) = .empty;
    defer log.deinit(std.testing.allocator);
    for ([_]*MockModel{ &llm, &diff }) |m| {
        m.has_log = true;
        m.var_log = &log;
    }
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
    };
    arb.rebalance();

    // Diffusion yields exactly the deficit (22 − 18 = 4 GiB kept) …
    try std.testing.expectEqual(@as(?u64, 4 << 30), diff.applied);
    // … and the LLM gets its full demand rather than being capped at 14 GiB.
    try std.testing.expectEqual(@as(?u64, 18 << 30), llm.applied);
    // SHRINK BEFORE GROW: diffusion must free before the LLM allocates into it,
    // or the LLM grows into a card the other model still occupies.
    try std.testing.expectEqual(@as(usize, 2), log.items.len);
    try std.testing.expectEqualStrings("diff", log.items[0].name);
    try std.testing.expectEqualStrings("llm", log.items[1].name);
}

test "Arbiter: an idle image model is NOT stomped when there is room for both" {
    // The counterpart guarantee: yielding is deficit-driven, so an image model
    // that fits alongside the LLM keeps every byte (the next image reloads none).
    var llm: MockModel = .{ .used = 10 << 30, .want = 10 << 30, .floor_b = 2 << 30 };
    var diff: MockModel = .{ .used = 6 << 30, .want = 6 << 30 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
    };
    arb.rebalance();
    try std.testing.expectEqual(@as(u64, 6 << 30), diff.used); // untouched
    try std.testing.expectEqual(@as(u64, 10 << 30), llm.used);
}

test "Arbiter: mid-image diffusion is never shrunk (its floor is its working set)" {
    // Soft residency: evicting mid-image would force per-step weight streaming.
    // A busy diffusion participant reports floor == usage, so even a starving LLM
    // cannot claw it back; the LLM keeps its own guaranteed share instead.
    var llm: MockModel = .{ .used = 16 << 30, .want = 20 << 30, .floor_b = 2 << 30 };
    var diff: MockModel = .{ .used = 9 << 30, .want = 9 << 30, .floor_b = 9 << 30, .is_busy = true };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
        .diff_active = true,
    };
    arb.rebalance();
    try std.testing.expectEqual(@as(u64, 9 << 30), diff.used); // held at its floor
    try std.testing.expectEqual(@as(?u64, 6 << 30), llm.applied); // down to its share
}

test "Arbiter.plan: the split handle bounds diffusion when both models are active" {
    // TODO #2 "split is not respected": under real contention the meter's split
    // handle is what divides the card. The LLM is guaranteed `llm_share` and
    // diffusion gets the rest — not "whatever the LLM happens to hold".
    var llm: MockModel = .{ .used = 20 << 30, .want = 20 << 30, .floor_b = 2 << 30 };
    var diff: MockModel = .{ .used = 0, .want = 12 << 30 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
        .diff_active = true,
    };
    const p = arb.plan().?;
    try std.testing.expectEqual(@as(u64, 6 << 30), p.llm); // guaranteed its share
    try std.testing.expectEqual(@as(u64, 16 << 30), p.diffusion); // the rest, 22 − 6
    try std.testing.expectEqual(@as(u64, 16 << 30), arb.diffusionBudget());

    // Move the handle: the division follows it, rather than following whatever the
    // LLM happens to be holding.
    arb.llm_share = 14 << 30;
    const p2 = arb.plan().?;
    try std.testing.expectEqual(@as(u64, 14 << 30), p2.llm);
    try std.testing.expectEqual(@as(u64, 8 << 30), p2.diffusion);
}

test "Arbiter.plan: a target never breaches a floor or exceeds a demand" {
    // Ceiling far below what either side wants: the floors win, and neither
    // target is inflated past the model's actual footprint.
    var llm: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .floor_b = 5 << 30 };
    var diff: MockModel = .{ .used = 4 << 30, .want = 4 << 30, .floor_b = 3 << 30 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 6 << 30, // less than either floor pair
        .llm_share = 1 << 30,
    };
    const p = arb.plan().?;
    try std.testing.expectEqual(@as(u64, 5 << 30), p.llm); // clamped up to its floor
    try std.testing.expectEqual(@as(u64, 3 << 30), p.diffusion); // clamped up to its floor
    try std.testing.expect(p.llm <= 8 << 30 and p.diffusion <= 4 << 30); // never above demand
}

test "Arbiter.plan: an unsatisfiable LLM target stays 0 rather than becoming 1" {
    // Regression guard on the new 1-byte floor. Diffusion is mid-image holding
    // MORE than the (just-lowered) ceiling, so the plan has nothing to give the
    // LLM. Flooring that to 1 byte would publish a real ceiling of 1 byte and the
    // LLM would migrate every layer to the host chasing it — the qwen3-32B
    // mass-offload bug. It must come out 0, which `settle` then declines.
    std.testing.log_level = .err; // the declined settle logs on purpose
    var llm: MockModel = .{ .used = 4 << 30, .want = 4 << 30, .floor_b = 0 };
    var diff: MockModel = .{ .used = 10 << 30, .want = 10 << 30, .floor_b = 10 << 30, .is_busy = true };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 8 << 30, // ceiling dragged BELOW what diffusion already holds
        .llm_share = 0,
    };
    const p = arb.plan().?;
    try std.testing.expectEqual(@as(u64, 0), p.llm);
    // Diffusion, by contrast, is floored to 1 — "hold nothing" is legitimate for a
    // pure cache — though its own busy floor keeps it where it is here.
    try std.testing.expectEqual(@as(u64, 10 << 30), p.diffusion);

    // And the LLM is left alone rather than stripped.
    arb.rebalance();
    try std.testing.expectEqual(@as(?u64, null), llm.applied);
    try std.testing.expectEqual(@as(u64, 4 << 30), llm.used);
}

test "Arbiter.plan: an idle image model can be asked to hold nothing" {
    // The counterpart: a starving LLM and an idle image model whose entire
    // residency is reclaimable. The target must reach it (1, not 0-as-no-op).
    var llm: MockModel = .{ .used = 20 << 30, .want = 30 << 30, .floor_b = 2 << 30 };
    var diff: MockModel = .{ .used = 4 << 30, .want = 4 << 30, .floor_b = 0 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
    };
    const p = arb.plan().?;
    try std.testing.expectEqual(@as(u64, 1), p.diffusion);
    arb.rebalance();
    // Everything but the 1-byte sentinel is gone (the real evict frees whole
    // weight buffers, so it lands at 0; the mock tracks the target exactly).
    try std.testing.expect(diff.used <= 1);
    // The LLM gets the whole ceiling (its 30 GiB demand exceeds it, so the ceiling
    // is what binds — not diffusion's former 4 GiB).
    try std.testing.expectEqual(@as(?u64, 22 << 30), llm.applied);
}

test "Arbiter.requestRoom: the LLM can reclaim from an idle image model" {
    // The reactive mirror of the pipeline's Reclaim hook. Because the two models
    // hold VRAM in separate device contexts, this is the ONLY way an LLM
    // allocation can reach diffusion's bytes.
    var llm: MockModel = .{ .used = 14 << 30, .want = 18 << 30, .floor_b = 2 << 30 };
    var diff: MockModel = .{ .used = 8 << 30, .want = 8 << 30 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
    };
    try std.testing.expectEqual(@as(u64, 3 << 30), arb.requestRoom(.llm, 3 << 30));
    try std.testing.expectEqual(@as(u64, 5 << 30), diff.used);
    // It frees NOW without publishing a new ceiling — `rebalance` stays the sole
    // owner of the steady-state target.
    try std.testing.expectEqual(@as(?u64, null), diff.cp.budgetTarget());

    // Capped by what the peer actually holds above its floor.
    diff.floor_b = 4 << 30;
    try std.testing.expectEqual(@as(u64, 1 << 30), arb.requestRoom(.llm, 8 << 30));
    try std.testing.expectEqual(@as(u64, 4 << 30), diff.used);
    // Nothing left to give.
    try std.testing.expectEqual(@as(u64, 0), arb.requestRoom(.llm, 8 << 30));
}

test "Arbiter.requestRoom: a busy peer is never touched, and a missing peer is 0" {
    std.testing.log_level = .err; // the decline logs on purpose
    var llm: MockModel = .{ .used = 14 << 30, .want = 18 << 30 };
    var diff: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .is_busy = true };
    var arb: Arbiter = .{ .llm = llm.participant(), .diffusion = diff.participant(), .limit = 22 << 30 };
    // Busy: its own worker owns that device context.
    try std.testing.expectEqual(@as(u64, 0), arb.requestRoom(.llm, 4 << 30));
    try std.testing.expectEqual(@as(u64, 8 << 30), diff.used);

    // No peer registered at all (CLI / studio-only).
    arb.diffusion = null;
    try std.testing.expectEqual(@as(u64, 0), arb.requestRoom(.llm, 4 << 30));
    // And the reverse direction works the same way.
    arb.diffusion = diff.participant();
    arb.llm = null;
    try std.testing.expectEqual(@as(u64, 0), arb.requestRoom(.diffusion, 4 << 30));
}

test "Participant.settle: a zero target is ignored, not clamped up to the floor" {
    // Regression: the qwen3-32B first-message mass-offload. A zero raw target
    // (uninitialized arbiter) used to be clamped UP to the committed-KV floor
    // (384 MiB mid-prefill) and published as a real ceiling — below the model's
    // un-evictable minimum, so the worker evicted every layer chasing it.
    std.testing.log_level = .err; // the skip logs on purpose; keep a passing run silent
    var m: MockModel = .{ .used = 20 << 30, .floor_b = 384 << 20 };
    m.participant().settle(0);
    try std.testing.expectEqual(@as(?u64, null), m.applied);
    try std.testing.expectEqual(@as(?u64, null), m.cp.budgetTarget());
}

test "Arbiter: uninitialized budgets (limit 0) never drive the LLM" {
    // Regression companion: with `setBudgets` never called (meter policy
    // early-returned on a failed VRAM query), the per-frame diffusion
    // queue-drained edge used to rebalance with limit 0 and publish the
    // floor-clamped garbage that `settle` now also rejects. The arbiter must
    // leave residency alone until it has real budgets.
    std.testing.log_level = .err; // the skipped rebalances warn on purpose
    var m: MockModel = .{ .used = 20 << 30, .want = 21 << 30, .floor_b = 384 << 20 };
    var arb: Arbiter = .{ .llm = m.participant() }; // limit/llm_share left 0
    try std.testing.expectEqual(@as(?Plan, null), arb.plan());
    arb.setDiffusionActive(false); // the empty-queue drain edge
    arb.setDiffusionActive(true);
    try std.testing.expectEqual(@as(?u64, null), m.applied);
    try std.testing.expectEqual(@as(?u64, null), m.cp.budgetTarget());

    // First real setBudgets takes over and drives normally again.
    arb.setDiffusionActive(false);
    arb.setBudgets(22 << 30, 6 << 30);
    try std.testing.expectEqual(@as(?u64, 21 << 30), m.applied); // its demand, ≤ limit
}

test "Arbiter: a busy LLM still yields (via its control point, not a direct apply)" {
    var m: MockModel = .{ .used = 20 << 30, .want = 20 << 30, .floor_b = 2 << 30, .is_busy = true };
    var arb: Arbiter = .{ .llm = m.participant(), .limit = 22 << 30, .llm_share = 6 << 30 };
    arb.setDiffusionActive(true);
    // The core bug fix: busy no longer means "decline". Nothing applied directly…
    try std.testing.expectEqual(@as(?u64, null), m.applied);
    // …but the target is published, so the LLM worker yields at its next token.
    try std.testing.expectEqual(@as(?u64, 6 << 30), m.cp.budgetTarget());
}

test "Arbiter.diffusionBudget: uninitialized budgets mean auto (0), not the 256 MiB floor" {
    // Regression: pure image-studio mode (no LLM session) never runs the meter
    // policy, so `limit` stays 0. diffusionBudget used to return
    // max(256 MiB, 0) — a hard 256 MiB pin budget that pinned a sliver of the
    // image model and evicted the rest. 0 is the pipeline's AUTO sentinel.
    std.testing.log_level = .err; // the auto fallback logs on purpose
    var arb: Arbiter = .{};
    try std.testing.expectEqual(@as(u64, 0), arb.diffusionBudget());

    // Still auto with an LLM registered but budgets unresolved.
    var m: MockModel = .{ .used = 20 << 30, .want = 20 << 30 };
    arb.llm = m.participant();
    try std.testing.expectEqual(@as(u64, 0), arb.diffusionBudget());

    // Real budgets: the room left once the LLM is at its planned target, floored
    // at 256 MiB.
    arb.limit = 22 << 30;
    arb.llm_share = 6 << 30;
    try std.testing.expectEqual(@as(u64, 2 << 30), arb.diffusionBudget()); // idle: limit − demand
    m.used = 22 << 30;
    m.want = 22 << 30;
    try std.testing.expectEqual(@as(u64, 256 << 20), arb.diffusionBudget()); // floored
}

test "ControlPoint: budget intent round-trips and clears" {
    var cp: ControlPoint = .{};
    // Default: no constraint.
    try std.testing.expectEqual(@as(?u64, null), cp.budgetTarget());

    cp.requestBudget(8 << 30);
    try std.testing.expectEqual(@as(?u64, 8 << 30), cp.budgetTarget());

    // Last write wins; persistent (peek does not consume it).
    cp.requestBudget(4 << 30);
    try std.testing.expectEqual(@as(?u64, 4 << 30), cp.budgetTarget());
    try std.testing.expectEqual(@as(?u64, 4 << 30), cp.budgetTarget());

    // A budget of 0 is a real target (evict everything), distinct from "no limit".
    cp.requestBudget(0);
    try std.testing.expectEqual(@as(?u64, 0), cp.budgetTarget());

    cp.clearBudget();
    try std.testing.expectEqual(@as(?u64, null), cp.budgetTarget());
}

test "ControlPoint: pause intent is observable (reserved, inert)" {
    var cp: ControlPoint = .{};
    try std.testing.expect(!cp.pausePending());
    cp.requestPause(true);
    try std.testing.expect(cp.pausePending());
    cp.requestPause(false);
    try std.testing.expect(!cp.pausePending());
}

test {
    std.testing.refAllDecls(@This());
}
