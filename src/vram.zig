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
/// look identical to the arbiter. Read methods (`usage`/`floor`/`busy`) may be
/// called from any thread; `applyBudget` mutates device residency and so is
/// only invoked on a thread that may bind this model's context — the idle path
/// (arbiter thread) or the model's own worker (via `pollAndApply`).
pub const Participant = struct {
    ctx: *anyopaque,
    control: *ControlPoint,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Device bytes this model currently holds.
        usage: *const fn (ctx: *anyopaque) u64,
        /// Bytes that cannot be evicted (LLM: committed KV; diffusion: working
        /// minimum). The arbiter never targets a model below its floor.
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

/// The single owner of "how much VRAM each model may hold." Replaces the ad-hoc
/// enter/exit/budget/reclaim hooks whose three code paths settled the LLM to
/// three disagreeing targets and no-op'd whenever it was busy. Every relevant
/// event (image-queue edge, generation start/stop, meter commit) calls one of
/// the mutators, which recomputes the authoritative target for each model and
/// drives it through `Participant.settle`.
///
/// Staging note: the LLM is a full participant now (it can settle live via its
/// control point). Diffusion is still a budget CONSUMER — it reads
/// `diffusionBudget()` at spawn for its `pin_budget` — until Stage 3 gives the
/// pipeline a live `giveUpToBudget`, at which point it becomes a `Participant`
/// too and `rebalance` drives both symmetrically.
pub const Arbiter = struct {
    llm: ?Participant = null,
    /// Total VRAM ceiling the models share (the meter's `limit` handle × card).
    limit: u64 = 0,
    /// The LLM's guaranteed floor under contention (the meter's `split` handle).
    llm_share: u64 = 0,
    /// Diffusion has queued/running work that wants VRAM (queue non-empty).
    diff_active: bool = false,
    /// Diffusion's current device residency (bytes). So an idle-but-resident
    /// image model isn't stomped when the LLM reclaims: while diffusion is
    /// inactive the LLM targets `limit - diff_used`, leaving its weights in
    /// place. Updated by the app on diffusion load/trim; 0 = holds nothing.
    diff_used: u64 = 0,

    /// Recompute + drive the LLM's authoritative residency target. Idempotent;
    /// safe to call on any event. When diffusion is ACTIVE the LLM yields down
    /// to its share (even mid-generation — the settle is published to its
    /// control point and applied at the next token). When diffusion is idle the
    /// LLM may use everything the image model isn't currently holding.
    ///
    /// With `limit == 0` the arbiter was never given budgets (`setBudgets`
    /// hasn't run, or the VRAM query behind it failed) — there is no real
    /// ceiling to drive toward, so driving anyway would publish garbage. Warn
    /// and leave residency alone; the next successful `setBudgets` rebalances.
    pub fn rebalance(self: *Arbiter) void {
        if (self.llm) |llm| {
            if (self.limit == 0) {
                std.log.warn("[vram] rebalance skipped: budgets uninitialized (meter policy never resolved — VRAM query failed?)", .{});
                return;
            }
            const target = if (self.diff_active) self.llm_share else self.limit -| self.diff_used;
            llm.settle(target);
        }
    }

    /// Diffusion queue started (`true`) / drained (`false`). Triggers a rebalance
    /// so the LLM starts yielding immediately, before the image model loads.
    pub fn setDiffusionActive(self: *Arbiter, active: bool) void {
        self.diff_active = active;
        self.rebalance();
    }

    /// Update diffusion's current residency (on image load / trim / unload).
    pub fn setDiffusionUsage(self: *Arbiter, used: u64) void {
        self.diff_used = used;
        self.rebalance();
    }

    /// Update the resolved ceiling + LLM floor (meter drag commit) and rebalance.
    pub fn setBudgets(self: *Arbiter, limit: u64, llm_share: u64) void {
        self.limit = limit;
        self.llm_share = llm_share;
        self.rebalance();
    }

    /// The resident-weight budget the next image may pin. When diffusion is
    /// active the LLM has committed to drop to `llm_share`, so diffusion may plan
    /// for `limit - llm_share` even while the LLM is still coming down (the VAE
    /// reclaim ladder covers the transient); otherwise it gets whatever the LLM
    /// isn't currently holding. Floored so a tiny budget still streams, never 0.
    ///
    /// With `limit == 0` the arbiter has no budgets (`setBudgets` never ran —
    /// there is no LLM session for the meter policy to resolve against, i.e.
    /// pure image-studio mode). Return 0 = the pipeline's AUTO sentinel (pin
    /// what fits live free VRAM), matching the pre-arbiter `vcBudget` behavior.
    /// Returning the 256 MiB floor here instead pinned a sliver of the image
    /// model and evicted/streamed the rest (the studio twin of the LLM
    /// zero-budget mass-offload bug).
    pub fn diffusionBudget(self: *const Arbiter) u64 {
        if (self.limit == 0) {
            std.log.debug("[vram] diffusion budget: arbiter uninitialized (no LLM/meter policy) → auto (pin what fits)", .{});
            return 0;
        }
        const min_budget: u64 = 256 << 20;
        const llm_committed = if (self.diff_active)
            self.llm_share
        else if (self.llm) |llm| llm.usage() else 0;
        return @max(min_budget, self.limit -| llm_committed);
    }
};

// --- tests -----------------------------------------------------------------

/// A mock model standing in for an LLM/diffusion session, recording the last
/// `applyBudget` it was driven to and whether it was applied directly.
const MockModel = struct {
    used: u64,
    floor_b: u64 = 0,
    is_busy: bool = false,
    applied: ?u64 = null, // last applyBudget target (null = never applied directly)
    cp: ControlPoint = .{},

    fn usageFn(ctx: *anyopaque) u64 {
        return mock(ctx).used;
    }
    fn floorFn(ctx: *anyopaque) u64 {
        return mock(ctx).floor_b;
    }
    fn busyFn(ctx: *anyopaque) bool {
        return mock(ctx).is_busy;
    }
    fn applyFn(ctx: *anyopaque, target: u64) void {
        mock(ctx).applied = target;
    }
    fn mock(ctx: *anyopaque) *MockModel {
        return @ptrCast(@alignCast(ctx));
    }
    const vtable: Participant.VTable = .{ .usage = usageFn, .floor = floorFn, .busy = busyFn, .applyBudget = applyFn };
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
    var m: MockModel = .{ .used = 20 << 30, .floor_b = 2 << 30 };
    var arb: Arbiter = .{ .llm = m.participant(), .limit = 22 << 30, .llm_share = 6 << 30 };

    // Diffusion starts → LLM yields to its share immediately.
    arb.setDiffusionActive(true);
    try std.testing.expectEqual(@as(?u64, 6 << 30), m.applied);
    // Diffusion may plan for limit - share even though the LLM still shows 20G used.
    try std.testing.expectEqual(@as(u64, 16 << 30), arb.diffusionBudget());

    // Diffusion drains → LLM may reclaim up to the whole limit.
    arb.setDiffusionActive(false);
    try std.testing.expectEqual(@as(?u64, 22 << 30), m.applied);

    // …but an idle-but-resident image model is left room (not stomped): the LLM
    // targets limit - diff_used instead of the whole limit.
    arb.setDiffusionUsage(5 << 30);
    try std.testing.expectEqual(@as(?u64, 17 << 30), m.applied);
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
    var m: MockModel = .{ .used = 20 << 30, .floor_b = 384 << 20 };
    var arb: Arbiter = .{ .llm = m.participant() }; // limit/llm_share left 0
    arb.setDiffusionActive(false); // the empty-queue drain edge
    arb.setDiffusionActive(true);
    arb.setDiffusionUsage(1 << 30);
    try std.testing.expectEqual(@as(?u64, null), m.applied);
    try std.testing.expectEqual(@as(?u64, null), m.cp.budgetTarget());

    // First real setBudgets takes over and drives normally again.
    arb.setDiffusionActive(false);
    arb.setBudgets(22 << 30, 6 << 30);
    try std.testing.expectEqual(@as(?u64, 21 << 30), m.applied); // limit − diff_used
}

test "Arbiter: a busy LLM still yields (via its control point, not a direct apply)" {
    var m: MockModel = .{ .used = 20 << 30, .floor_b = 2 << 30, .is_busy = true };
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
    var m: MockModel = .{ .used = 20 << 30 };
    arb.llm = m.participant();
    try std.testing.expectEqual(@as(u64, 0), arb.diffusionBudget());

    // Real budgets: back to limit − committed, floored at 256 MiB.
    arb.limit = 22 << 30;
    arb.llm_share = 6 << 30;
    try std.testing.expectEqual(@as(u64, 2 << 30), arb.diffusionBudget()); // idle: limit − usage
    m.used = 22 << 30;
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
