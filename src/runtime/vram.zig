//! Cross-model VRAM coordination.
//!
//! The GUI runs an LLM and a diffusion model on the same device and must
//! arbitrate VRAM between them (a big image push must be able to migrate LLM
//! layers to the host, and reclaim them when it finishes). The pieces here are
//! backend/model-agnostic and live in the library so both frontends can use
//! them (the diffusion CLI and tp-llm have a single model and need no arbiter,
//! but the primitives are shared).
//!
//! Design constraint that shapes everything: a CUDA context is bound
//! per-thread, each model's device residency may only be mutated on the
//! worker thread that bound its context. So the arbiter never touches the GPU
//! directly. It computes a desired residency budget per model and publishes it
//! to that model's `ControlPoint`; the model's own worker observes the intent
//! at a safe boundary (between LLM tokens / diffusion steps) and applies it on
//! its own thread. That eliminates cross-thread device races by construction,
//! the failure mode of the ad-hoc `imageVramEnter`/`settleLlm` hooks this
//! replaces.

const std = @import("std");

/// How much of the card this process may commit, and why.
///
/// The rule:
///
///   requested = total - limit          how much of the card must NOT be ours
///   reserve   = max(requested, foreign)  what actually stays out of our hands
///   ours      = total - reserve          cap on our WHOLE footprint
///   tracked   = ours - untracked         cap on what our allocators count
///
/// `reserve` is a max, not a subtraction, and other processes count TOWARDS it.
/// The handle asks that at least N bytes of the card are not this process's; the
/// desktop holding more than N already satisfies that, so the handle stops
/// binding and every setting collapses to `total - foreign`. That is intended.
///
/// The alternative, `budget = limit - system` with `system` the residual
/// `device_used - our_tracked`, enforces `card_used <= limit` but puts an
/// unreliable measurement alone on the right-hand side: any under-reading of
/// `system` inflates the budget directly, we promote layers to fill it, and the
/// first big allocation OOMs. It IS under-read, because the policy resolves right
/// after a load, before our own CUDA context, JIT'd modules and library
/// workspaces exist. Taking the larger of what the user asked for and what other
/// processes hold means an under-measured `foreign` cannot inflate anything; it
/// falls back to the user's figure.
///
/// This reserve is NOT the forward's transient headroom, and must not be read as
/// it. It bounds RESIDENCY. Once it is satisfied by other processes the budget
/// permits filling the rest of the card, so the room a forward's scratch needs
/// has to be kept by whoever grows into that budget -- see `Participant.headroom`
/// and the K/V growth path, which subtract `largestAllocation` for exactly this.
///
/// The cap is on our WHOLE footprint because physical usage is
/// `tracked + our untracked + foreign`. Capping only `tracked` leaves the untracked part
/// (~1.1 GiB on a 3090 with one LLM resident) with nowhere to live, so the card
/// overcommits by exactly that much and the next batched allocation fails. The caller
/// passes a HIGH-WATER `untracked` so a cold reading of ~0 cannot re-inflate the budget
/// mid-session.
pub const Reserve = struct {
    /// Card bytes that must stay out of our hands.
    bytes: u64,
    /// What the user asked to keep free (`total - limit`).
    requested: u64,
    /// What other processes actually hold.
    foreign: u64,
    /// Cap on our whole footprint (tracked + untracked).
    ours: u64,
    /// Cap on what our allocators count, what the arbiter budgets.
    tracked: u64,
    /// Our untracked overhead, as folded into the cap.
    untracked: u64,

    /// "card 24101 * want 361 free, others hold 2064 -> reserve 2064 * ours <= 22037
    ///  (untracked 1104) -> tracked <= 20933 MiB"
    pub fn render(self: Reserve, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        w.print("want {d} free, others hold {d} -> reserve {d} · ours <= {d} (untracked {d}) -> tracked <= {d} MiB", .{
            self.requested >> 20, self.foreign >> 20,    self.bytes >> 20,
            self.ours >> 20,      self.untracked >> 20,  self.tracked >> 20,
        }) catch return "?";
        return w.buffered();
    }
};

/// The user's ceiling. The CLI passes an absolute figure (`--vram-budget`), the
/// GUI a fraction of the card (the meter's limit handle); both resolve HERE so
/// the two frontends cannot drift apart on what "limit" means.
pub const Limit = union(enum) {
    /// No user ceiling: the reserve is whatever other processes hold.
    none,
    /// Whole-card ceiling in bytes.
    card_bytes: u64,
    /// Whole-card ceiling as a fraction of the card, the GUI meter's handle.
    fraction: f32,
    /// A cap on OUR OWN tracked residency rather than on total card usage. This
    /// is what `tp-llm --vram-budget <GiB>` has always meant ("cap device memory
    /// for the split planner"), and it is kept as a distinct variant rather than
    /// converted so the CLI flag's documented meaning does not silently change.
    /// The physical reserve still applies on top: a cap larger than the card can
    /// safely hold is clamped down.
    ours_bytes: u64,

    /// The whole-card ceiling this limit implies (the card itself when the limit
    /// only constrains our own share).
    fn cardCeiling(self: Limit, total: u64) u64 {
        return switch (self) {
            .none, .ours_bytes => total,
            .card_bytes => |b| @min(b, total),
            .fraction => |f| @min(@as(u64, @intFromFloat(f * @as(f64, @floatFromInt(total)))), total),
        };
    }
};

/// One coherent measurement of the card. Every field MUST come from the same
/// sampling pass, mixing a stale total with live component reads is what made
/// the old status-bar residual bounce at frame rate.
pub const Card = struct {
    total: u64,
    /// Bytes charged to OTHER processes. Pass `null` when the driver exposes no
    /// per-process data; the resolver then degrades to treating everything that
    /// is not our tracked bytes as foreign, which is the old behaviour and is
    /// conservative in the right direction.
    foreign: ?u64,
    /// Our whole footprint on the card (per-process reading). Ignored when
    /// `foreign` is null.
    ours_total: u64 = 0,
    /// What our allocators count.
    ours_tracked: u64 = 0,
    /// Whole-card bytes in use, all processes. Only needed for the degraded path.
    device_used: u64 = 0,
};

/// Apply the rule above. `untracked_floor` is the caller's high-water mark for
/// our own unattributed VRAM; pass 0 the first time and feed back
/// `Reserve.untracked` thereafter so a cold reading cannot re-inflate the cap.
pub fn resolve(limit: Limit, card: Card, untracked_floor: u64) Reserve {
    const requested = card.total -| limit.cardCeiling(card.total);
    const foreign = card.foreign orelse (card.device_used -| card.ours_tracked);
    const untracked = @max(
        if (card.foreign != null) card.ours_total -| card.ours_tracked else 0,
        untracked_floor,
    );
    const reserve = @max(requested, foreign);
    const ours = card.total -| reserve;
    // A `.ours_bytes` limit caps our tracked share on top of the physical reserve;
    // whichever binds first wins.
    var tracked = ours -| untracked;
    if (limit == .ours_bytes) tracked = @min(tracked, limit.ours_bytes);
    return .{
        .bytes = reserve,
        .requested = requested,
        .foreign = foreign,
        .ours = ours,
        .tracked = tracked,
        .untracked = untracked,
    };
}

/// An itemized device-residency figure in bytes. Both the PREDICTION (what a
/// model says it will need) and the MEASUREMENT (what it actually holds) use this
/// same shape, so the two can be printed side by side and differenced.
///
/// Itemized rather than a single number on purpose. Every residency bug in this
/// subsystem has been ONE TERM SILENTLY MISSING from a scalar estimate, the LM
/// head, the embeddings, the RoPE tables, the dequant staging, and a scalar
/// offers nothing to check against reality, so each was found only by noticing
/// layers stranded on the host. With line items, a wrong estimate is visible the
/// first time a model loads (see `logResidency`).
pub const Bytes = struct {
    /// Model parameters, whether or not they have been uploaded yet.
    weights: u64 = 0,
    /// KV cache / recurrent state at the current capacity.
    kv: u64 = 0,
    /// Everything sized from the batch or context rather than the parameter
    /// count: activations, logits, RoPE tables, dequant staging.
    scratch: u64 = 0,
    /// Ours but unattributed, what the process holds on the card that our
    /// allocators never counted: the CUDA context and JIT'd modules, cuBLASLt /
    /// cuDNN workspaces, UI textures. Only ever MEASURED (it has no predictable
    /// size), and only when the driver exposes per-process data.
    untracked: u64 = 0,

    pub fn total(self: Bytes) u64 {
        return self.weights + self.kv + self.scratch + self.untracked;
    }

    pub fn plus(self: Bytes, o: Bytes) Bytes {
        return .{
            .weights = self.weights + o.weights,
            .kv = self.kv + o.kv,
            .scratch = self.scratch + o.scratch,
            .untracked = self.untracked + o.untracked,
        };
    }

    /// "weights 17025 + kv 1234 + scratch 370 = 18629 MiB" (untracked shown only
    /// when nonzero, i.e. on a measurement). Writes into `buf`; returns the slice.
    pub fn render(self: Bytes, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        w.print("weights {d} + kv {d} + scratch {d}", .{ self.weights >> 20, self.kv >> 20, self.scratch >> 20 }) catch return "?";
        if (self.untracked != 0) w.print(" + untracked {d}", .{self.untracked >> 20}) catch return "?";
        w.print(" = {d} MiB", .{self.total() >> 20}) catch return "?";
        return w.buffered();
    }
};

/// Log a participant's predicted need against what it actually holds, itemized,
/// with the difference spelled out. The whole point is that a prediction which
/// disagrees with reality is loud rather than silent: the arbiter turns `need`
/// into a residency ceiling, so a term missing from the prediction becomes a cap
/// below what the model actually costs, and the model then offloads layers to
/// obey it while the card sits half empty.
///
/// `have.untracked` is not in `need` by design (it cannot be predicted), so it is
/// reported separately rather than folded into the delta.
pub fn logResidency(who: []const u8, need_b: Bytes, have: Bytes) void {
    var nb: [160]u8 = undefined;
    var hb: [160]u8 = undefined;
    std.log.info("[vram] {s} predicted: {s}", .{ who, need_b.render(&nb) });
    // A prediction of FULL residency is only checkable against a model that IS
    // fully resident. Two legitimate reasons it might not be: weights upload
    // lazily on the first forward (cold), or layers have been offloaded to the
    // host (squeezed). Comparing in either state would cry wolf, so report how
    // much is resident and skip the verdict.
    if (have.weights < need_b.weights) {
        std.log.info("[vram] {s} actual:    {s}  ({d}/{d} MiB of weights resident — not fully loaded, no verdict)", .{
            who, have.render(&hb), have.weights >> 20, need_b.weights >> 20,
        });
        return;
    }
    const delta: i64 = @as(i64, @intCast(have.total())) - @as(i64, @intCast(need_b.total()));
    std.log.info("[vram] {s} actual:    {s}  (vs predicted: {c}{d} MiB{s})", .{
        who,                                  have.render(&hb),
        @as(u8, if (delta < 0) '-' else '+'), @abs(delta) >> 20,
        // Loud on purpose: the arbiter turns `need` into a residency ceiling, so a
        // prediction below reality caps the model under its own footprint and
        // strands layers on the host. One layer is the scale that matters.
        if (@abs(delta) > (256 << 20)) " <-- PREDICTION IS WRONG" else "",
    });
}

/// A cooperative control point between a model's compute WORKER thread and an
/// external COORDINATOR (the app-level `VramArbiter`; later also a pause UI).
/// One per participant, embedded in the model's session. The coordinator
/// publishes *intents* from any thread; the worker *polls* them at its own safe
/// boundaries and acts on its own thread. Nothing here touches the device.
///
/// Two intents:
///   - `budget`, the desired device-residency ceiling (bytes) the worker
///     should settle toward. Persistent + last-write-wins (it is also the
///     ongoing growth ceiling, not a one-shot), `unconstrained` = no limit.
///   - `pause`, reserved for the upcoming pause feature (park the worker at
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
    /// stays in effect, settle ops are idempotent, so re-observing a satisfied
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
/// is only invoked on a thread that may bind this model's context, the idle path
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
        /// `limit − llm_usage == diff_usage`, its own residency back, i.e. a
        /// guaranteed no-op, every time. See `Arbiter.plan`.
        demand: *const fn (ctx: *anyopaque) u64,
        /// Free bytes this model needs ABOVE its steady-state `demand` for the
        /// transient buffers a forward pass allocates. Not residency: room that
        /// must stay unclaimed under the model's ceiling so it can actually reach
        /// full residency.
        ///
        /// This exists because a ceiling equal to `demand` is a ceiling the
        /// promote path cannot fill. `residency.promoteBack` refuses to bring a
        /// layer back unless its cost PLUS that transient fits underneath, so the
        /// last layers stay stranded on the host forever, and no allocation ever
        /// fails, so nothing reactive fires either. How many layers strand is
        /// `transient / bytes-per-layer`, which on a 1-bit 27B (~60 MiB/layer)
        /// against a ~1 GiB image-prefill buffer is most of the back half of the
        /// model, running on the CPU. Diffusion reports 0: it has its own reclaim
        /// ladder and re-uploads per image.
        headroom: *const fn (ctx: *anyopaque) u64,
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
        /// Give up EVERYTHING, keeping the work this model is queued to do.
        /// Optional: null means "no such rung".
        ///
        /// `applyBudget` can only shed what a model treats as CACHE — for
        /// diffusion, its weight LRU. What survives is the resident pipeline
        /// itself: its context, its library workspaces, its activation scratch.
        /// No budget target can reach those, so a peer that has "yielded
        /// everything" can still be holding GiBs, and the asker's only remaining
        /// option was to run on the CPU beside VRAM nothing was using.
        ///
        /// Only meaningful where the whole residency is recoverable work: an idle
        /// image model reloads its pipeline for the next image. The LLM has no
        /// such rung (its weights are pinned and its yield is layer migration,
        /// which `applyBudget` already does), so it leaves this null.
        releaseAll: ?*const fn (ctx: *anyopaque) void = null,
        /// Has a `releaseAll` been accepted but not yet enacted? Optional: null
        /// means this model always frees inside the `releaseAll` call, so there
        /// is never anything to wait for.
        ///
        /// A model whose residency has one writer (diffusion: the UI thread owns
        /// the session pointer, every reader loads it unlocked) cannot free from
        /// the asker's thread, so `releaseAll` queues the work and returns with
        /// nothing freed. Without this hook the asker cannot tell that apart from
        /// a peer that had nothing to give, and an asker that is ABOUT TO
        /// ALLOCATE has to treat both as failure. See `pendingRelease`.
        releasePending: ?*const fn (ctx: *anyopaque) bool = null,
    };

    pub fn usage(self: Participant) u64 {
        return self.vtable.usage(self.ctx);
    }
    pub fn demand(self: Participant) u64 {
        return @max(self.vtable.demand(self.ctx), self.vtable.usage(self.ctx));
    }
    /// Steady-state footprint plus the transient a forward allocates on top of
    /// it: the smallest ceiling under which this model can reach full residency.
    /// This, not `demand`, is what a plan must leave a model.
    pub fn claim(self: Participant) u64 {
        return self.demand() +| self.vtable.headroom(self.ctx);
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
    /// measurement), and clamping it up to the floor would manufacture a real,
    /// and typically unreachable, ceiling out of garbage, evicting the whole
    /// model (the qwen3-32B first-message mass-offload bug). A coordinator that
    /// wants a model to yield everything it can targets a small nonzero budget;
    /// the floor clamp keeps that safe.
    pub fn settle(self: Participant, target: u64) Settle {
        if (target == 0) {
            std.log.debug("[vram] settle skipped: zero target (would clamp to floor {d} MiB)", .{self.floor() >> 20});
            return .skipped;
        }
        const clamped = @max(target, self.floor());
        if (clamped != target)
            std.log.debug("[vram] settle target {d} MiB below the committed-KV floor; raised to {d} MiB", .{ target >> 20, clamped >> 20 });
        self.control.requestBudget(clamped);
        if (self.busy()) return .deferred;
        self.vtable.applyBudget(self.ctx, clamped);
        return if (self.usage() > clamped) .short else .applied;
    }

    /// Worker-side: at a safe boundary, enact any pending published ceiling on
    /// this (the model's own) thread. Cheap no-op when already settled.
    pub fn pollAndApply(self: Participant) void {
        if (self.control.budgetTarget()) |t| self.vtable.applyBudget(self.ctx, t);
    }
};

/// What a `settle` actually achieved. A settle used to be fire-and-forget, which
/// hid the one outcome that needs handling: a yield the peer DECLINED.
/// `Diffuser.giveUpToBudget` tryLocks its residency mutex and returns 0 rather
/// than block a foreign thread, so a settle that loses that race leaves both
/// models in exactly the state the plan just rejected, and nothing re-runs it
/// until the next external edge (a meter drag, a queue edge). `.short` is what
/// makes that recoverable.
pub const Settle = enum {
    /// Applied on this thread; the model is at or under the target.
    applied,
    /// The model is busy, so the target is published only; its worker enacts it
    /// at the next safe boundary. Expected, not a failure.
    deferred,
    /// No real target (0). Nothing published, nothing applied.
    skipped,
    /// Applied, and the model is STILL above the target: it declined, or it has
    /// nothing further it is allowed to release. Worth one bounded retry.
    short,
};

/// Which model a cross-model request is on behalf of.
pub const Side = enum { llm, diffusion };

/// A coherent residency target for BOTH models, computed in one pass.
pub const Plan = struct {
    /// Residency CEILING published to the LLM, how much it MAY hold.
    llm: u64,
    diffusion: u64,
    /// What the LLM actually NEEDS: its steady-state demand PLUS the transient a
    /// forward allocates on top of it (`Participant.claim`). Differs from the
    /// ceiling in the uncontended case, where the ceiling is deliberately the
    /// whole unclaimed budget. Diffusion's allowance is computed against this:
    /// the room the LLM does not need is claimable, whereas the room it is merely
    /// permitted to grow into is a soft cap it yields when an image queue starts.
    llm_need: u64,
    /// Which arm of `plan` produced this. Logged, because "why is my model
    /// offloaded" is almost always "a different arm fired than you assumed".
    arm: Arm = .uncontended,

    pub const Arm = enum {
        /// Both fit; nobody gets a growth-blocking ceiling.
        uncontended,
        /// Over budget with BOTH models working, the split handle decides.
        split,
        /// Over budget, only diffusion working, the idle LLM yields the deficit.
        diffusion_only,
        /// Over budget, diffusion idle, its cache yields to the LLM.
        llm_only,
    };
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
/// structural defect behind the "diffusion never yields to the LLM" report,
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
    /// Genuinely external state, the queue is not derivable from residency,
    /// unlike the old `diff_used` mirror of `diffusion.usage()`, which was
    /// hand-updated from three call sites and went stale between them.
    diff_active: bool = false,
    /// The last `rebalance` applied a target a model did not reach (see
    /// `Settle.short`). Cleared by a rebalance in which everything landed.
    unsettled: bool = false,
    /// Consecutive retries of an unsettled plan. Bounded, because `.short` also
    /// covers "this model has nothing left it is allowed to release", which no
    /// amount of retrying fixes; reset by every external event.
    retries: u8 = 0,

    /// Retries beyond which an unsettled plan is treated as the best available
    /// outcome rather than a lost race. A declined yield is resolved by the
    /// holder finishing what it is doing, which is frames away, not minutes.
    const max_retries = 8;

    /// Compute both models' residency targets from one snapshot of (demand,
    /// floor) plus the meter's ceiling/split. Pure: reads participants, mutates
    /// nothing. Returns null when there is no real ceiling to plan against
    /// (`limit == 0`: `setBudgets` never ran, or the VRAM query behind it
    /// failed), driving residency from that would publish garbage, which is how
    /// the qwen3-32B first-message mass-offload happened.
    ///
    /// Policy, in priority order:
    ///
    ///  1. No contention (both demands fit under the ceiling), everyone
    ///     stays fully resident. Covers the single-model case too: an absent
    ///     participant contributes 0 demand and 0 floor.
    ///  2. Diffusion active (something queued or generating), diffusion
    ///     gets the room, except that the meter's split handle guarantees the
    ///     LLM `llm_share` no matter what. The LLM yields even mid-generation:
    ///     the target is published to its control point and applied at its next
    ///     token boundary.
    ///  3. Diffusion idle, its residency is *opportunistic cache* (the next
    ///     image re-uploads what it needs), so the LLM has priority and
    ///     diffusion yields exactly the deficit: `limit − llm_claim`, never
    ///     below its own floor, never above what it wants. This is the direction
    ///     that never worked.
    ///
    /// Each side is clamped into `[floor, claim]`, so a target can neither evict
    /// something un-evictable nor inflate a model past what it can use.
    ///
    /// The LLM's side is sized by `Participant.claim` (demand PLUS the transient a
    /// forward allocates), never by demand alone. Demand is an estimate of
    /// steady-state residency; a ceiling set to it is one the promote path cannot
    /// fill, which is not an estimate error the model can recover from, it is
    /// layers left on the CPU with nothing to signal it.
    pub fn plan(self: *const Arbiter) ?Plan {
        if (self.limit == 0) return null;

        // `claim`, not `demand`, everywhere below. A ceiling equal to demand is a
        // ceiling `residency.promoteBack` cannot fill: it needs the forward's
        // transient to fit UNDER the ceiling too, so it stops promoting a whole
        // transient's worth of layers early and they run on the CPU forever, with
        // no allocation failing and nothing reactive to fire. See
        // `Participant.VTable.headroom`.
        const l_claim = if (self.llm) |p| p.claim() else 0;
        const l_flr = if (self.llm) |p| p.floor() else 0;
        const d_flr = if (self.diffusion) |p| p.floor() else 0;
        // An ACTIVE queue is entitled to the complement of the LLM's guaranteed
        // share whether or not we can size the image model yet, this is the whole
        // meaning of the split handle. Floored rather than measured because at the
        // queue-start edge the pipeline is not loaded (`usage` is 0) and the
        // estimate behind `demand` reads checkpoint file sizes, which can fail; a
        // demand of 0 there would read as "no contention" and leave the LLM holding
        // the entire card for the image model to load into.
        // An IDLE image model demands only what it is actually holding. Its
        // `demand` is an estimate from CHECKPOINT FILE SIZES, which is right when
        // a queue is about to load the pipeline and badly wrong otherwise: a
        // merely-configured diffuser reported ~17 GiB it did not hold, every plan
        // read as contended, and arm (4) then computed `d = limit - the LLM's
        // need` and `l = limit - d`, pinning its ceiling to EXACTLY that need.
        // With no headroom above demand the first transient prefill allocation
        // (measured: a 1.1 GiB dequant buffer) had nowhere to go and OOM'd, which
        // offloaded layers, which is the eviction churn this whole path exists to
        // prevent. When a queue does start, `setDiffusionActive` re-plans and the
        // estimate (plus the entitlement below) takes over.
        const d_raw = if (self.diffusion) |p| (if (self.diff_active) p.demand() else p.usage()) else 0;
        const d_dem: u64 = if (self.diff_active) @max(d_raw, self.limit -| self.llm_share) else d_raw;
        // What diffusion actually WANTS, as opposed to what an active queue is
        // entitled to. The entitlement above deliberately inflates to the LLM's
        // complement so a queue that hasn't loaded yet still reads as contention;
        // but where we hand diffusion only its need (arm 3), inflating it would
        // evict far more of the LLM than the image model will ever use. Fall back
        // to the entitlement when the estimate is genuinely unavailable.
        const d_want: u64 = if (d_raw != 0) d_raw else d_dem;

        const l_busy = if (self.llm) |p| p.busy() else false;

        // ONE rule, four cases. What every arm keeps true: the two ceilings
        // SUM TO AT MOST `limit`, so the cap holds even with both models running.
        //
        // "Active" means WORKING RIGHT NOW, not merely loaded. Between messages a
        // chat session is fair game: the image model may take what it needs and the
        // LLM gives it up, because promote/migrate is fast and in practice only the
        // deficit moves. That is a deliberate product call, the split is a
        // contention rule, not a standing reservation, so an idle model never holds
        // VRAM that a working one could use.
        var out: Plan = if (l_claim + d_dem <= self.limit) blk: {
            // (1) Uncontended, both fit. Neither gets a growth-blocking ceiling:
            // the LLM may use everything diffusion does not need.
            //
            // Emphatically NOT the LLM's own claim. `Participant.demand` is
            // `max(demand, usage)`, so handing that back pins the ceiling to what
            // the model already holds, and the first byte of KV growth then trips
            // `settleTo`'s `deviceUsed() > target` and offloads a layer. Measured
            // on a 12B with nothing else resident: budget 21395 MiB, target 8224,
            // usage 8227, one layer pushed to the host with 13.3 GiB free.
            //
            // The slack goes to the LLM rather than being split: it is the side
            // that grows continuously (KV) and the side where yielding costs an
            // interactive stall. Diffusion is offered its demand, which is all it
            // can use, and `diffusionBudget` still reports the room the LLM does
            // not NEED, so an image model can still load into it.
            break :blk .{ .llm = self.limit -| d_dem, .diffusion = d_dem, .llm_need = l_claim, .arm = .uncontended };
        } else if (self.diff_active and l_busy) blk: {
            // (2) BOTH working and over budget, the only case the split decides.
            const l = clamp(self.llm_share, l_flr, l_claim);
            break :blk .{ .llm = l, .diffusion = clamp(self.limit -| l, d_flr, d_dem), .llm_need = l, .arm = .split };
        } else if (self.diff_active) blk: {
            // (3) Only diffusion is working. It takes what it NEEDS (the `d_want`
            // clamp) rather than everything it is allowed, so the idle LLM yields
            // just the deficit instead of being evicted wholesale, that is what
            // keeps the bouncing small. `l_flr` (committed KV) is never taken.
            const d = clamp(self.limit -| l_flr, d_flr, d_want);
            break :blk .{ .llm = @max(self.limit -| d, l_flr), .diffusion = d, .llm_need = @min(l_claim, self.limit -| d), .arm = .diffusion_only };
        } else blk: {
            // (4) Diffusion idle: its residency is pure cache, so the LLM takes
            // what it needs and diffusion gives the rest back (down to its floor,
            // which is 0 unless an image is mid-flight).
            const d = clamp(self.limit -| l_claim, d_flr, d_dem);
            break :blk .{ .llm = @max(self.limit -| d, l_flr), .diffusion = d, .llm_need = @min(l_claim, self.limit -| d), .arm = .llm_only };
        };
        // A CEILING is not a grant: never clamp it down to demand. Demand is an
        // estimate of steady-state residency and carries no allowance for the
        // transient buffers a forward pass allocates, so a ceiling pinned at demand
        // guarantees the next spike OOMs. `llm_need` above still reports the real
        // need, which is what diffusion's allowance is computed against.
        // "Hold nothing" is a legitimate target for DIFFUSION, its residency is
        // pure cache, but 0 is also the sentinel the enactment paths read as "no
        // target, do nothing". Ask for 1 byte instead; the incremental evict then
        // frees `usage − 1`, i.e. everything.
        //
        // The LLM deliberately gets NO such floor. A 0 there means the plan had
        // nothing to give it (the ceiling dragged below what a mid-image diffusion
        // holds, or the split handle at zero with no committed KV yet to floor it),
        // and turning that into a real 1-byte ceiling would offload every layer to
        // the host chasing it, the qwen3-32B mass-offload bug. `settle` rejects a
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
    /// released yet, i.e. into a full card. Shrink first, grow second. Ordering
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
        {
            // Log the INPUTS and the arm, not just the outcome: every residency
            // surprise so far has been "a different arm fired than you assumed",
            // and the arm is a function of numbers you otherwise cannot see.
            const lu = if (self.llm) |q| q.usage() else 0;
            const du = if (self.diffusion) |q| q.usage() else 0;
            std.log.info("[vram] plan {t}: limit {d} · llm need {d} (holds {d}) -> ceiling {d} · diff need {d} (holds {d}) -> ceiling {d}{s}", .{
                p.arm,        self.limit >> 20,
                p.llm_need >> 20, lu >> 20, p.llm >> 20,
                (self.limit -| p.llm_need) >> 20, du >> 20, p.diffusion >> 20,
                if (self.diff_active) " · image queue ACTIVE" else "",
            });
        }
        const llm_shrinks = if (self.llm) |q| p.llm < q.usage() else false;
        var l_out: Settle = .skipped;
        var d_out: Settle = .skipped;
        if (llm_shrinks) {
            if (self.llm) |q| l_out = q.settle(p.llm);
            if (self.diffusion) |q| d_out = q.settle(p.diffusion);
        } else {
            if (self.diffusion) |q| d_out = q.settle(p.diffusion);
            if (self.llm) |q| l_out = q.settle(p.llm);
        }
        self.noteOutcome(l_out, d_out);
    }

    /// Record whether the plan actually landed. A `.short` diffusion settle is
    /// the interesting one: `giveUpToBudget` declines rather than blocks, so the
    /// LLM is then left growing into room that was never freed. Nothing else in
    /// the system notices, which is why it is recorded here.
    fn noteOutcome(self: *Arbiter, l_out: Settle, d_out: Settle) void {
        if (l_out != .short and d_out != .short) {
            self.unsettled = false;
            self.retries = 0;
            return;
        }
        self.unsettled = true;
        std.log.info("[vram] plan not fully enacted (llm {t}, diffusion {t}) — retry {d}/{d}", .{
            l_out, d_out, self.retries + 1, max_retries,
        });
    }

    /// Re-run a plan that did not land. Called from the frontend's frame loop;
    /// returns whether it retried, so a caller can throttle. Bounded: `.short`
    /// also means "nothing left to release", which retrying cannot fix.
    pub fn retryUnsettled(self: *Arbiter) bool {
        if (!self.unsettled or self.retries >= max_retries) return false;
        self.retries += 1;
        self.rebalance();
        return true;
    }

    /// MID-TURN growth poll, called from the busy LLM's own worker at a
    /// boundary, not from an external edge. Returns whether it raised anything.
    ///
    /// `rebalance` runs on edges only: a queue starting or draining, a meter
    /// drag, the turn's own idle->busy edge. Nothing re-plans WHILE a turn runs,
    /// so a turn that began squeezed (layers on the host) stayed squeezed for the
    /// whole of it however much the image model handed back meanwhile — and the
    /// prefill it stayed squeezed through is the slow half. The edges that do
    /// fire are not enough on their own: clearing the image model retracts the
    /// participant without re-planning, and `retryUnsettled` is driven by the
    /// frame loop, which emits no frames behind a prefill.
    ///
    /// GROWTH only, which is not timidity: mid-turn `usage()` carries the
    /// transient buffers the forward just allocated, so a full plan from here
    /// reads the LLM as fatter than it is and would migrate layers to the HOST in
    /// the middle of the prefill this exists to speed up. Shrinking stays with
    /// the edges and with the reactive `requestRoom` rung, which run when the
    /// numbers are steady or when a peer actually needs the room.
    ///
    /// Publishes, never applies: the caller is the worker, so `settle` defers to
    /// that worker's next boundary by construction and no foreign thread binds
    /// the model's context. The peer is not settled either; the promote path
    /// asks it for card space itself (`requestRoom`) if the raise needs bytes it
    /// is still holding.
    pub fn pollLlmGrowth(self: *Arbiter) bool {
        const q = self.llm orelse return false;
        if (!q.busy()) return false; // idle: `rebalance` owns the steady state
        const p = self.plan() orelse return false;
        const cur = q.control.budgetTarget() orelse q.usage();
        if (p.llm <= cur) return false;
        std.log.info("[vram] mid-turn: LLM ceiling {d} -> {d} MiB (plan {t}); its worker promotes at the next boundary", .{
            cur >> 20, p.llm >> 20, p.arm,
        });
        _ = q.settle(p.llm);
        return true;
    }

    /// Diffusion queue started (`true`) / drained (`false`). Triggers a rebalance
    /// so the LLM starts yielding immediately, before the image model loads.
    pub fn setDiffusionActive(self: *Arbiter, active: bool) void {
        self.diff_active = active;
        self.retries = 0; // a real event: a fresh budget of retries
        self.rebalance();
    }

    /// Update the resolved ceiling + LLM floor (meter drag commit) and rebalance.
    pub fn setBudgets(self: *Arbiter, limit: u64, llm_share: u64) void {
        self.limit = limit;
        self.llm_share = llm_share;
        self.retries = 0;
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
    /// work around by offloading its own layers to the CPU, or, when the
    /// allocation wasn't a layer it could migrate, by failing the turn outright
    /// with DeviceOutOfMemory.
    ///
    /// Only an IDLE peer is touched: the apply binds the peer's device context,
    /// which its own worker would otherwise be using. Callable from either
    /// model's worker thread, so it deliberately does NOT publish a new ceiling
    /// to the peer's control point, `rebalance` stays the single owner of the
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
        var freed = used -| peer.usage();
        std.log.info("[vram] {t} asked for {d} MiB: peer yielded {d} MiB ({d}→{d} MiB resident)", .{
            side, bytes >> 20, freed >> 20, used >> 20, peer.usage() >> 20,
        });
        if (freed < bytes) freed += self.releasePeer(peer, side, bytes - freed);
        return freed;
    }

    /// Last rung of `requestRoom`: the incremental yield came up short, so ask an
    /// idle peer to drop its residency ENTIRELY (`VTable.releaseAll`).
    ///
    /// This is the difference between "the image model gave back what it could"
    /// and "the LLM fits". `applyBudget` bottoms out at the peer's irreducible
    /// footprint — context, workspaces, activation scratch — which is exactly the
    /// GiB or two that leaves the last layers of a large LLM stranded on the CPU,
    /// running at a fraction of device speed beside VRAM that nothing is using.
    ///
    /// Three conditions, and each one is load-bearing:
    ///   still short   an asker that got what it needed never pays this.
    ///   not busy      mid-image eviction is what soft residency exists to avoid.
    ///   queue IDLE    `busy` is only "generating right now". Releasing between two
    ///                 queued images would hand back VRAM the next one reloads
    ///                 seconds later, which is the bouncing this system is built to
    ///                 prevent. A contended queue is what the split handle decides.
    fn releasePeer(self: *Arbiter, peer: Participant, side: Side, still_short: u64) u64 {
        const rel = peer.vtable.releaseAll orelse return 0;
        // The ASKER has to be working. `promoteBack` also runs on an IDLE model —
        // every image that finishes rebalances, which hands the idle LLM a big
        // ceiling that it then applies on the spot — and that ask is speculative:
        // it is filling a ceiling it has no work for. Honouring it here threw away
        // the image model's entire pipeline between two images, which the next
        // image reloaded from disk seconds later. A model that is actually
        // generating asks again at its next boundary, and THAT is the ask worth a
        // peer's whole residency.
        const asker = (switch (side) {
            .llm => self.llm,
            .diffusion => self.diffusion,
        }) orelse return 0;
        if (!asker.busy()) {
            std.log.debug("[vram] {t} is idle: promoting speculatively, not releasing the peer", .{side});
            return 0;
        }
        const peer_queued = switch (side) {
            .llm => self.diff_active, // the peer IS diffusion
            .diffusion => false, // an LLM turn shows up as `busy`
        };
        if (peer_queued or peer.busy()) return 0;
        const used = peer.usage();
        if (used == 0) return 0;
        rel(peer.ctx);
        const freed = used -| peer.usage();
        // A peer may enact this on its OWN thread at its own safe point rather
        // than here (diffusion does: its session pointer has one writer, so the
        // free belongs to the UI thread). Then nothing is free yet and the asker
        // must not act as though it were — it picks the room up at its next
        // boundary poll, which is a frame away, not a turn away.
        if (freed == 0) {
            std.log.info("[vram] {t} still {d} MiB short: asked the peer to release its {d} MiB residency; it drops at its next safe point", .{
                side, still_short >> 20, used >> 20,
            });
            return 0;
        }
        std.log.info("[vram] {t} still {d} MiB short: peer RELEASED its whole residency, {d} MiB ({d}→{d} MiB) — it reloads on its next job", .{
            side, still_short >> 20, freed >> 20, used >> 20, peer.usage() >> 20,
        });
        return freed;
    }

    /// `side`'s peer is sitting on a release it accepted and has not enacted yet:
    /// its current residency, i.e. the bytes about to come back. Null when
    /// nothing is pending, so there is nothing to wait for.
    ///
    /// This is the other half of `requestRoom` for a caller that is about to
    /// allocate. `requestRoom` reports the bytes freed BY THE TIME IT RETURNS,
    /// which for a deferred release is zero — and a zero that means "any moment
    /// now" is not the same answer as a zero that means "the peer has nothing",
    /// though it reads identically. `residency.promoteBack` can conflate them
    /// safely (it allocated nothing and re-asks at its next boundary); an OOM
    /// ladder cannot, because for it the next step is failing the caller's turn.
    ///
    /// Deliberately a QUERY and not a wait: the enacting thread may need a lock
    /// the asker is holding to reach this arbiter at all, so only the caller can
    /// know where it is safe to block. See `app.awaitDeferredRelease`.
    pub fn pendingRelease(self: *const Arbiter, side: Side) ?u64 {
        const peer = (switch (side) {
            .llm => self.diffusion,
            .diffusion => self.llm,
        }) orelse return null;
        const p = peer.vtable.releasePending orelse return null;
        if (!p(peer.ctx)) return null;
        return peer.usage();
    }

    /// The resident-weight budget the next image may pin: the room the plan
    /// leaves once the LLM is at its planned target, so it agrees with what
    /// `rebalance` is driving the LLM toward. One formula, not two: reading the LLM's
    /// live usage in one place and its planned share in another makes them disagree.
    /// Deliberately the *allowance* rather than `plan.diffusion`,
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
        // Against the LLM's NEED, not its ceiling: uncontended, the ceiling is the
        // whole unclaimed budget (so the LLM can grow into free VRAM instead of
        // offloading), but the room it does not NEED is exactly what an image model
        // may still claim. Using the ceiling here would offer diffusion the floor
        // and it could never load.
        return @max(min_budget, self.limit -| p.llm_need);
    }
};

fn clamp(v: u64, lo: u64, hi: u64) u64 {
    return @min(@max(v, lo), @max(lo, hi));
}

/// Compare byte figures at MiB granularity: a `fraction` limit goes through f32,
/// so byte-exact expectations are meaningless (0.985 of a 24101 MiB card lands
/// half a MiB either side depending on rounding).
fn expectMiB(expected: u64, actual: u64) !void {
    const a = actual >> 20;
    if (a != expected and a != expected -| 1 and a != expected + 1) {
        std.debug.print("expected ~{d} MiB, found {d} MiB\n", .{ expected, a });
        return error.TestExpectedEqual;
    }
}

test "vram.resolve: other processes count towards the reserve, so the handle stops binding" {
    const total: u64 = 24101 << 20;
    // The measured case: 2064 MiB of desktop, 1104 MiB of our own untracked VRAM.
    const r = resolve(.{ .fraction = 0.985 }, .{
        .total = total,
        .foreign = 2064 << 20,
        .ours_total = 21637 << 20,
        .ours_tracked = 20533 << 20,
    }, 0);
    try expectMiB(361, r.requested); // 1.5% of the card
    try expectMiB(2064, r.bytes); // foreign already keeps more than the request out
    try expectMiB(1104, r.untracked);
    try expectMiB(22037, r.ours);
    try expectMiB(20933, r.tracked);
    // The whole point: ours + reserve never exceeds the card.
    try std.testing.expect(r.ours + r.bytes <= total);
    try std.testing.expect(r.tracked + r.untracked + r.bytes <= total);
}

test "vram.resolve: an under-read of foreign still cannot inflate the cap past the request" {
    const total: u64 = 24101 << 20;
    // foreign reads as 0 (the cold case, before anything else has allocated).
    // The user's figure has to remain the floor, which is why the request is a
    // term in the sum rather than something foreign can wash out.
    const r = resolve(.{ .fraction = 0.985 }, .{
        .total = total,
        .foreign = 0,
        .ours_total = 0,
        .ours_tracked = 0,
    }, 0);
    try expectMiB(361, r.requested);
    try expectMiB(361, r.bytes); // degrades to the user's figure, never below it
    try std.testing.expect(r.bytes >= r.requested);
}

test "vram.resolve: the user's reserve wins when other processes are small" {
    const total: u64 = 24101 << 20;
    const r = resolve(.{ .fraction = 0.75 }, .{
        .total = total,
        .foreign = 200 << 20,
        .ours_total = 1000 << 20,
        .ours_tracked = 1000 << 20,
    }, 0);
    try expectMiB(6025, r.requested);
    try expectMiB(6025, r.bytes); // requested wins
    try std.testing.expectEqual(@as(u64, 0), r.untracked);
    try std.testing.expect(r.ours + r.bytes <= total);
}

test "vram.resolve: the untracked high-water floor cannot be re-inflated by a cold read" {
    const total: u64 = 24101 << 20;
    // Cold: our process has barely allocated, so the live untracked read is ~0.
    // Without the floor this hands back a cap 1.1 GiB too generous, we promote
    // layers to fill it, and the first batched allocation OOMs.
    const cold = resolve(.{ .fraction = 0.985 }, .{
        .total = total,
        .foreign = 2064 << 20,
        .ours_total = 3299 << 20,
        .ours_tracked = 3299 << 20,
    }, 1104 << 20);
    try expectMiB(1104, cold.untracked);
    try expectMiB(20933, cold.tracked);
    // The property under test, independent of the arithmetic: a cold read must
    // land on the SAME cap as a warm one, not a more generous one.
    const warm = resolve(.{ .fraction = 0.985 }, .{
        .total = total,
        .foreign = 2064 << 20,
        .ours_total = 21637 << 20,
        .ours_tracked = 20533 << 20,
    }, 0);
    try std.testing.expectEqual(warm.tracked, cold.tracked);
}

test "vram.resolve: no NVML degrades to the residual, and never overcommits" {
    const total: u64 = 24101 << 20;
    // foreign == null: everything that is not our tracked bytes counts as foreign,
    // which folds our own untracked into the reserve. Conservative, and exactly
    // the pre-NVML behaviour.
    const r = resolve(.{ .fraction = 0.985 }, .{
        .total = total,
        .foreign = null,
        .ours_tracked = 20533 << 20,
        .device_used = 23701 << 20,
    }, 0);
    try expectMiB(3168, r.foreign);
    try std.testing.expectEqual(@as(u64, 0), r.untracked);
    try std.testing.expect(r.tracked + r.bytes <= total);
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
    /// Transient a forward needs above `want` (real LLMs: ~1 GiB during an image
    /// prefill). 0 in most tests so their arithmetic stays readable.
    hdr: u64 = 0,
    floor_b: u64 = 0,
    /// Refuse `applyBudget` outright, standing in for `giveUpToBudget`'s tryLock
    /// decline. The target is recorded but residency does not move.
    declines: bool = false,
    /// Residency `applyBudget` CANNOT reach, however low the target: the image
    /// pipeline's context, library workspaces and activation scratch, none of
    /// which live in the weight cache it evicts from. Distinct from `floor_b`,
    /// which is what the model REPORTS as un-evictable; this is what it actually
    /// bottoms out at, and the gap between the two is the whole reason
    /// `releaseAll` exists.
    irreducible: u64 = 0,
    /// Set when the `releaseAll` rung fired.
    released: bool = false,
    /// Enact `releaseAll` LATER, on the model's own thread, as diffusion does
    /// (its session pointer has one writer, so the free belongs to the UI thread).
    /// Nothing is free when the rung returns.
    release_defers: bool = false,
    /// A deferred release has been accepted and not yet enacted, the flag the
    /// real participant reports through `VTable.releasePending`. `enactRelease`
    /// is the mock's stand-in for the UI thread getting to it.
    release_pending: bool = false,
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
    fn headroomFn(ctx: *anyopaque) u64 {
        return mock(ctx).hdr;
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
        if (m.declines) return; // the target was seen and ignored, as a tryLock decline is
        // Real participants only ever move toward the target: shrink to it, or
        // grow up to their demand. Mirror that so `used` stays believable.
        if (target < m.used)
            m.used = @max(@max(target, m.floor_b), m.irreducible)
        else
            m.used = @min(target, @max(m.want, m.used));
    }
    fn releaseFn(ctx: *anyopaque) void {
        const m = mock(ctx);
        m.released = true;
        if (m.declines) return;
        if (m.release_defers) {
            m.release_pending = true;
            return;
        }
        m.used = 0; // the whole pipeline goes, irreducible part included
    }
    fn releasePendingFn(ctx: *anyopaque) bool {
        return mock(ctx).release_pending;
    }
    /// The deferred free finally happening on the model's own thread.
    fn enactRelease(self: *MockModel) void {
        self.release_pending = false;
        self.used = 0;
    }
    fn mock(ctx: *anyopaque) *MockModel {
        return @ptrCast(@alignCast(ctx));
    }
    const vtable: Participant.VTable = .{
        .usage = usageFn,
        .demand = demandFn,
        .headroom = headroomFn,
        .floor = floorFn,
        .busy = busyFn,
        .applyBudget = applyFn,
    };
    fn participant(self: *MockModel) Participant {
        return .{ .ctx = self, .control = &self.cp, .vtable = &vtable };
    }
    /// A participant that also exposes the drop-everything rung, as diffusion
    /// does and the LLM deliberately does not.
    const vtable_rel: Participant.VTable = .{
        .usage = usageFn,
        .demand = demandFn,
        .headroom = headroomFn,
        .floor = floorFn,
        .busy = busyFn,
        .applyBudget = applyFn,
        .releaseAll = releaseFn,
        .releasePending = releasePendingFn,
    };
    fn releasable(self: *MockModel) Participant {
        return .{ .ctx = self, .control = &self.cp, .vtable = &vtable_rel };
    }
};

test "Participant.settle: idle applies now, busy defers to control point" {
    var m: MockModel = .{ .used = 8 << 30, .floor_b = 1 << 30 };

    // Idle: applied directly AND published.
    _ = m.participant().settle(3 << 30);
    try std.testing.expectEqual(@as(?u64, 3 << 30), m.applied);
    try std.testing.expectEqual(@as(?u64, 3 << 30), m.cp.budgetTarget());

    // Busy: published only; NOT applied directly (would race the worker's context).
    m.applied = null;
    m.is_busy = true;
    _ = m.participant().settle(2 << 30);
    try std.testing.expectEqual(@as(?u64, null), m.applied);
    try std.testing.expectEqual(@as(?u64, 2 << 30), m.cp.budgetTarget());

    // The busy worker later enacts it on its own thread.
    m.participant().pollAndApply();
    try std.testing.expectEqual(@as(?u64, 2 << 30), m.applied);
}

test "Participant.settle: target is clamped up to the floor" {
    var m: MockModel = .{ .used = 8 << 30, .floor_b = 4 << 30 };
    _ = m.participant().settle(1 << 30); // below floor
    try std.testing.expectEqual(@as(?u64, 4 << 30), m.applied);
    try std.testing.expectEqual(@as(?u64, 4 << 30), m.cp.budgetTarget());
}

test "Arbiter: diffusion active drives the LLM down to its share; idle frees it" {
    // The LLM wants the whole ceiling; no diffusion participant registered.
    var m: MockModel = .{ .used = 20 << 30, .want = 22 << 30, .floor_b = 2 << 30 };
    var arb: Arbiter = .{ .llm = m.participant(), .limit = 22 << 30, .llm_share = 6 << 30 };

    // Diffusion starts -> LLM yields to its share immediately.
    arb.setDiffusionActive(true);
    try std.testing.expectEqual(@as(?u64, 6 << 30), m.applied);
    // Diffusion may plan for limit − share even though the LLM only just started
    // coming down (the VAE reclaim ladder covers the transient).
    try std.testing.expectEqual(@as(u64, 16 << 30), arb.diffusionBudget());

    // Diffusion drains -> LLM may reclaim up to the whole limit.
    arb.setDiffusionActive(false);
    try std.testing.expectEqual(@as(?u64, 22 << 30), m.applied);
}

test "Arbiter: an IDLE resident image model yields to the LLM (TODO #1)" {
    // The reported bug. Diffusion holds 8 GiB of a 22 GiB budget and is idle; the
    // LLM wants 18 GiB. Old behaviour: the LLM was settled to limit − 8 = 14 GiB
    // (offloading layers to the CPU) and diffusion was then offered
    // limit − 14 = 8 GiB, exactly what it already held, so it never freed a byte.
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

    // Diffusion yields exactly the deficit (22 − 18 = 4 GiB kept) ...
    try std.testing.expectEqual(@as(?u64, 4 << 30), diff.applied);
    // ... and the LLM gets its full demand rather than being capped at 14 GiB.
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

    // The LLM here is IDLE, so the split does not bind: diffusion is given what it
    // needs (9) and the LLM keeps the remainder (22 - 9 = 13) rather than being
    // squeezed to its 6 GiB share for memory the image model will not use.
    try std.testing.expectEqual(@as(?u64, 13 << 30), llm.applied);

    // Once the LLM is working too, both want the card and the split decides. A
    // BUSY participant is not applied directly, the target is published and its
    // worker enacts it at the next token boundary (see the busy-yield test).
    llm.is_busy = true;
    arb.rebalance();
    try std.testing.expectEqual(@as(?u64, 6 << 30), llm.cp.budgetTarget());
    try std.testing.expectEqual(@as(u64, 9 << 30), diff.used); // still never shrunk
}

test "Arbiter.plan: an IDLE llm yields the deficit to a working diffuser" {
    // Deliberate policy: the split is a contention rule, not a standing
    // reservation. Between messages the image model may take what it needs and the
    // LLM gives it up, promote/migrate is fast and only the deficit moves.
    var llm: MockModel = .{ .used = 18 << 30, .want = 18 << 30, .floor_b = 2 << 30 }; // not busy
    var diff: MockModel = .{ .used = 0, .want = 9 << 30 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
        .diff_active = true,
    };
    const p = arb.plan().?;
    // Diffusion gets what it NEEDS (9), not the whole card and not `limit - share`.
    try std.testing.expectEqual(@as(u64, 9 << 30), p.diffusion);
    // The idle LLM yields exactly the deficit: 22 - 9 = 13, well above its 2 GiB
    // floor. It is NOT crushed to the floor, and NOT held at its 6 GiB share.
    try std.testing.expectEqual(@as(u64, 13 << 30), p.llm);
    try std.testing.expect(p.llm + p.diffusion <= arb.limit);

    // The moment the LLM starts a turn, the split takes over and it reclaims.
    llm.is_busy = true;
    try std.testing.expectEqual(@as(u64, 6 << 30), arb.plan().?.llm);
}

test "Arbiter.plan: uncontended, the LLM may grow into free VRAM" {
    // The regression this rule exists for: a ceiling equal to the LLM's demand is
    // a ceiling equal to what it already holds (`Participant.demand` maxes with
    // usage), so the next KV byte trips `settleTo` and offloads a layer into an
    // otherwise-empty card. Measured on a 12B: budget 21395 MiB, target 8224,
    // usage 8227, 13.3 GiB free, 1/48 layers pushed to the host.
    var llm: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .floor_b = 1 << 30 };
    var arb: Arbiter = .{ .llm = llm.participant(), .limit = 21 << 30, .llm_share = 6 << 30 };
    const p = arb.plan().?;
    try std.testing.expectEqual(@as(u64, 21 << 30), p.llm); // room to grow, not 8
    try std.testing.expectEqual(@as(u64, 8 << 30), p.llm_need); // but need is unchanged
    // Diffusion is still offered everything the LLM does not NEED, so an image
    // model can load even though the LLM's ceiling is the whole limit.
    try std.testing.expectEqual(@as(u64, 13 << 30), arb.diffusionBudget());
}

test "Arbiter.plan: an LLM ceiling always leaves room for the forward's transient" {
    // The stranded-layer bug. Arms 3 and 4 used to compute diffusion's allowance
    // as `limit - llm_demand` and then hand the LLM back `limit - that`, i.e.
    // EXACTLY its own demand. `residency.promoteBack` will not promote a layer
    // unless its cost plus the forward's transient fits under the ceiling, so a
    // ceiling at bare demand strands `transient / bytes-per-layer` layers on the
    // host permanently: no allocation fails, so nothing reactive fires, and the
    // model runs its prefill on the CPU while the image model sits idle.
    //
    // Numbers from the reported session: a 1-bit 27B (~6 GiB resident, ~60 MiB a
    // layer) with a ~1 GiB image-prefill transient, next to a resident krea2.
    const gib = 1 << 30;
    var llm: MockModel = .{ .used = 4 * gib, .want = 6 * gib, .hdr = gib, .floor_b = 512 << 20 };
    var diff: MockModel = .{ .used = 18 * gib, .want = 18 * gib, .floor_b = 0 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 21 * gib,
        .llm_share = 15 * gib,
    };
    const p = arb.plan().?;
    // Arm 4 (diffusion idle): the ceiling covers demand AND the transient, so the
    // promote loop can reach full residency. At bare demand it would be 6 GiB and
    // the last ~17 layers could never come back.
    try std.testing.expectEqual(Plan.Arm.llm_only, p.arm);
    try std.testing.expectEqual(@as(u64, 7 * gib), p.llm);
    try std.testing.expectEqual(@as(u64, 7 * gib), p.llm_need);
    // Diffusion gives up exactly that deficit and keeps the rest as cache.
    try std.testing.expectEqual(@as(u64, 14 * gib), p.diffusion);
    try std.testing.expect(p.llm + p.diffusion <= arb.limit);

    // Same guarantee once the LLM is working and the split binds (arm 2): the
    // share is what divides the card, but the clamp above it is the CLAIM, so a
    // share generous enough for full residency is not silently cut back to demand.
    llm.is_busy = true;
    arb.diff_active = true;
    const p2 = arb.plan().?;
    try std.testing.expectEqual(Plan.Arm.split, p2.arm);
    try std.testing.expectEqual(@as(u64, 7 * gib), p2.llm); // clamped to the claim, not to 6
}

test "Arbiter: a declined yield is recorded and retried, not lost" {
    // `Diffuser.giveUpToBudget` tryLocks and returns 0 rather than block a foreign
    // thread. A fire-and-forget settle that lost that race left the LLM growing
    // into room nobody freed, with no event scheduled to try again.
    std.testing.log_level = .err; // the retry logs on purpose
    const gib = 1 << 30;
    var llm: MockModel = .{ .used = 4 * gib, .want = 6 * gib, .floor_b = 0, .name = "llm" };
    var diff: MockModel = .{ .used = 18 * gib, .want = 18 * gib, .declines = true, .name = "diff" };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 21 * gib,
        .llm_share = 15 * gib,
    };
    arb.rebalance();
    try std.testing.expectEqual(@as(u64, 18 * gib), diff.used); // declined
    try std.testing.expect(arb.unsettled);

    // The frame loop retries, and the moment the peer stops declining it lands.
    try std.testing.expect(arb.retryUnsettled());
    diff.declines = false;
    try std.testing.expect(arb.retryUnsettled());
    try std.testing.expectEqual(@as(u64, 15 * gib), diff.used);
    try std.testing.expect(!arb.unsettled);
    // Nothing to retry once it landed.
    try std.testing.expect(!arb.retryUnsettled());
}

test "Arbiter.retryUnsettled: a peer with nothing left to give does not spin" {
    // `.short` also covers "already at my floor", which no retry can fix. The
    // budget must run out rather than rebalancing every frame forever.
    std.testing.log_level = .err;
    const gib = 1 << 30;
    var llm: MockModel = .{ .used = 4 * gib, .want = 6 * gib };
    var diff: MockModel = .{ .used = 18 * gib, .want = 18 * gib, .declines = true };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 21 * gib,
        .llm_share = 15 * gib,
    };
    arb.rebalance();
    var n: usize = 0;
    while (arb.retryUnsettled()) n += 1;
    try std.testing.expectEqual(@as(usize, Arbiter.max_retries), n);
    // A real event restores the budget (the situation genuinely changed).
    arb.setDiffusionActive(true);
    arb.setDiffusionActive(false);
    try std.testing.expect(arb.retryUnsettled());
}

test "Arbiter.plan: ceilings never sum past the limit" {
    // The hard guarantee behind the meter's limit handle: whatever the arm, the
    // two models may not be told they can collectively exceed it.
    var llm: MockModel = .{ .used = 4 << 30, .want = 18 << 30, .floor_b = 1 << 30 };
    var diff: MockModel = .{ .used = 0, .want = 9 << 30 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
    };
    for ([_]bool{ false, true }) |active| {
        arb.diff_active = active;
        for ([_]u64{ 4 << 30, 18 << 30, 30 << 30 }) |want| {
            llm.want = want;
            const p = arb.plan().?;
            // `plan` floors diffusion at 1 BYTE, the "hold nothing" sentinel, so
            // the enactment path can tell it apart from "no target". That is a
            // marker, not an allocation, so it does not count against the cap.
            const d: u64 = if (p.diffusion <= 1) 0 else p.diffusion;
            try std.testing.expect(p.llm + d <= arb.limit);
        }
    }
}

test "Arbiter.plan: the split handle bounds diffusion when both models are active" {
    // TODO #2 "split is not respected": under real contention the meter's split
    // handle is what divides the card. The LLM is guaranteed `llm_share` and
    // diffusion gets the rest, not "whatever the LLM happens to hold".
    // BOTH must be working for the split to apply, an idle LLM instead yields the
    // deficit to a working diffuser (see the arm-3 test below).
    var llm: MockModel = .{ .used = 20 << 30, .want = 20 << 30, .floor_b = 2 << 30, .is_busy = true };
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
    // LLM would migrate every layer to the host chasing it, the qwen3-32B
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
    // Diffusion, by contrast, is floored to 1, "hold nothing" is legitimate for a
    // pure cache, though its own busy floor keeps it where it is here.
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
    // is what binds, not diffusion's former 4 GiB).
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
    // It frees NOW without publishing a new ceiling, `rebalance` stays the sole
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
    // Regression: a first-message mass-offload. Clamping a zero raw target (an
    // uninitialized arbiter) UP to the committed-KV floor publishes it as a real
    // ceiling below the model's un-evictable minimum, so the worker evicts every
    // layer chasing it.
    std.testing.log_level = .err; // the skip logs on purpose; keep a passing run silent
    var m: MockModel = .{ .used = 20 << 30, .floor_b = 384 << 20 };
    _ = m.participant().settle(0);
    try std.testing.expectEqual(@as(?u64, null), m.applied);
    try std.testing.expectEqual(@as(?u64, null), m.cp.budgetTarget());
}

test "Arbiter: uninitialized budgets (limit 0) never drive the LLM" {
    // Regression companion: with `setBudgets` never called (the meter policy
    // early-returns on a failed VRAM query), the per-frame diffusion queue-drained
    // edge rebalances with limit 0 and publishes floor-clamped garbage. The arbiter
    // must leave residency alone until it has real budgets.
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
    // Uncontended, so the ceiling is the whole limit (nothing else claims any of
    // it) rather than the LLM's demand, a ceiling equal to demand is a ceiling
    // equal to current usage, which blocks KV growth and offloads layers into
    // free VRAM. `plan` arm (1).
    try std.testing.expectEqual(@as(?u64, 22 << 30), m.applied);
}

test "Arbiter: a busy LLM still yields (via its control point, not a direct apply)" {
    var m: MockModel = .{ .used = 20 << 30, .want = 20 << 30, .floor_b = 2 << 30, .is_busy = true };
    var arb: Arbiter = .{ .llm = m.participant(), .limit = 22 << 30, .llm_share = 6 << 30 };
    arb.setDiffusionActive(true);
    // Busy does not mean "decline". Nothing applied directly...
    try std.testing.expectEqual(@as(?u64, null), m.applied);
    // ...but the target is published, so the LLM worker yields at its next token.
    try std.testing.expectEqual(@as(?u64, 6 << 30), m.cp.budgetTarget());
}

test "Arbiter.requestRoom: a DEFERRED release reports only what is actually free" {
    // Diffusion enacts the rung on its own thread (the UI thread owns its session
    // pointer), so when `requestRoom` returns, nothing has been freed yet. It must
    // report 0 rather than the bytes the peer intends to give: `promoteBack` sizes
    // its next promote against what it can SEE, and a phantom credit here would
    // have it allocate into VRAM that is still occupied.
    var llm: MockModel = .{ .used = 6 << 30, .want = 20 << 30, .is_busy = true };
    var diff: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .irreducible = 3 << 30, .release_defers = true };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.releasable(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
    };
    // 5 GiB of weight cache trimmed now; the pipeline itself drops a frame later.
    try std.testing.expectEqual(@as(u64, 5 << 30), arb.requestRoom(.llm, 8 << 30));
    try std.testing.expect(diff.released); // asked
    try std.testing.expectEqual(@as(u64, 3 << 30), diff.used); // not yet enacted
}

test "Arbiter.pendingRelease: a deferred release is distinguishable from an empty peer" {
    // The 0 above is why this exists. A caller that is about to ALLOCATE has to
    // know whether that 0 means "nothing is coming" or "3 GiB, in a moment": one
    // is a failed turn, the other is a turn that waits a frame and succeeds. On a
    // 31B beside an idle krea2 the difference was a DeviceOutOfMemory on a 16 MiB
    // buffer, with the peer's 1444 MiB landing in the next log line.
    var llm: MockModel = .{ .used = 6 << 30, .want = 20 << 30, .is_busy = true };
    var diff: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .irreducible = 3 << 30, .release_defers = true };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.releasable(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
    };
    try std.testing.expectEqual(@as(?u64, null), arb.pendingRelease(.llm)); // nothing asked yet
    _ = arb.requestRoom(.llm, 8 << 30);
    // Pending, and worth the bytes the peer is still holding.
    try std.testing.expectEqual(@as(?u64, 3 << 30), arb.pendingRelease(.llm));
    diff.enactRelease();
    try std.testing.expectEqual(@as(?u64, null), arb.pendingRelease(.llm)); // landed: stop waiting
    try std.testing.expectEqual(@as(u64, 0), diff.used);
}

test "Arbiter.pendingRelease: null for a peer that frees inside the rung, and with no peer" {
    // A participant that enacts `releaseAll` synchronously (and the LLM, which has
    // no such rung at all) leaves `releasePending` null, so a waiting caller must
    // not wait: the bytes it was going to wait for are already in hand.
    var llm: MockModel = .{ .used = 6 << 30, .want = 20 << 30, .is_busy = true };
    var diff: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .irreducible = 3 << 30 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.releasable(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
    };
    _ = arb.requestRoom(.llm, 8 << 30);
    try std.testing.expect(diff.released);
    try std.testing.expectEqual(@as(u64, 0), diff.used); // freed inside the rung
    try std.testing.expectEqual(@as(?u64, null), arb.pendingRelease(.llm));
    // The other direction has no `releaseAll` (nor `releasePending`) to report.
    try std.testing.expectEqual(@as(?u64, null), arb.pendingRelease(.diffusion));
}

test "Arbiter.requestRoom: an IDLE asker never releases the peer" {
    // Regression: the image model was unloading between images with nothing else
    // running. Each drained queue rebalances, the idle LLM is handed a big ceiling
    // and applies it on the spot, and `promoteBack` then asks for card room to
    // fill it. That ask is speculative — no turn is running — but it was reaching
    // the release rung and dropping the whole pipeline, which the next image
    // reloaded from disk.
    var llm: MockModel = .{ .used = 6 << 30, .want = 20 << 30, .is_busy = false };
    var diff: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .irreducible = 3 << 30 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.releasable(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
    };
    _ = arb.requestRoom(.llm, 6 << 30);
    try std.testing.expect(!diff.released);
    try std.testing.expectEqual(@as(u64, 3 << 30), diff.used); // trimmed, still loaded

    // The same ask from a RUNNING turn is the one worth the peer's residency.
    llm.is_busy = true;
    _ = arb.requestRoom(.llm, 6 << 30);
    try std.testing.expect(diff.released);
    try std.testing.expectEqual(@as(u64, 0), diff.used);
}

test "Arbiter.requestRoom: an idle image model is RELEASED when trimming it is not enough" {
    // The reported symptom: the LLM ends up running layers on the CPU beside an
    // idle image model that "already yielded". It had — of its weight cache. The
    // 3 GiB underneath (pipeline context, workspaces, activation scratch) is not
    // cache, no budget target reaches it, and `promoteBack` just breaks and
    // leaves the rest of the model on the host.
    var llm: MockModel = .{ .used = 6 << 30, .want = 20 << 30, .is_busy = true }; // a turn is running
    var diff: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .irreducible = 3 << 30 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.releasable(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
    };

    const freed = arb.requestRoom(.llm, 6 << 30);
    try std.testing.expect(diff.released);
    try std.testing.expectEqual(@as(u64, 8 << 30), freed); // everything, not the 5 GiB of cache
    try std.testing.expectEqual(@as(u64, 0), diff.used);
}

test "Arbiter.requestRoom: the release rung is a LAST resort, not a default" {
    // Satisfied by the incremental yield: the pipeline stays, because reloading it
    // for the next image costs seconds and nothing needed those bytes.
    {
        var llm: MockModel = .{ .used = 6 << 30, .want = 20 << 30, .is_busy = true };
        var diff: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .irreducible = 1 << 30 };
        var arb: Arbiter = .{ .llm = llm.participant(), .diffusion = diff.releasable(), .limit = 22 << 30 };
        try std.testing.expectEqual(@as(u64, 2 << 30), arb.requestRoom(.llm, 2 << 30));
        try std.testing.expect(!diff.released);
        try std.testing.expectEqual(@as(u64, 6 << 30), diff.used);
    }
    // Images QUEUED (not generating this instant): releasing here hands back VRAM
    // the next image reloads seconds later, the bouncing this system exists to
    // prevent. Contention between two working models is the split handle's call.
    {
        var llm: MockModel = .{ .used = 6 << 30, .want = 20 << 30, .is_busy = true };
        var diff: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .irreducible = 3 << 30 };
        var arb: Arbiter = .{
            .llm = llm.participant(),
            .diffusion = diff.releasable(),
            .limit = 22 << 30,
            .diff_active = true,
        };
        _ = arb.requestRoom(.llm, 6 << 30);
        try std.testing.expect(!diff.released);
        try std.testing.expectEqual(@as(u64, 3 << 30), diff.used); // trimmed only
    }
    // A peer with no such rung (the LLM: pinned weights, migration is its yield)
    // is asked and left alone.
    {
        var llm: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .irreducible = 3 << 30 };
        var diff: MockModel = .{ .used = 2 << 30, .want = 8 << 30, .is_busy = true }; // an image is generating
        var arb: Arbiter = .{ .llm = llm.participant(), .diffusion = diff.participant(), .limit = 22 << 30 };
        _ = arb.requestRoom(.diffusion, 6 << 30);
        try std.testing.expect(!llm.released);
        try std.testing.expectEqual(@as(u64, 3 << 30), llm.used);
    }
}

test "Arbiter.pollLlmGrowth: a mid-turn poll raises a ceiling the edges left stale" {
    // The turn starts with an image queued, so the split handle pins the LLM to
    // its share and its prefill runs with layers on the host. Then the peer stops
    // holding VRAM without an edge reaching the arbiter: the image model is
    // cleared (`freeDiffuser` retracts the participant and clears `diff_active`
    // without re-planning), or the queue drained while the UI thread, the only
    // caller of `retryUnsettled`, was asleep behind a prefill that emits no
    // frames. Nothing re-plans until the turn ENDS, so the whole slow half runs
    // squeezed against VRAM nobody holds.
    var llm: MockModel = .{ .used = 6 << 30, .want = 20 << 30, .floor_b = 2 << 30, .is_busy = true };
    var diff: MockModel = .{ .used = 8 << 30, .want = 8 << 30, .floor_b = 0 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
    };
    arb.setDiffusionActive(true); // an image is queued: arm (2), both working
    try std.testing.expectEqual(@as(?u64, 6 << 30), llm.cp.budgetTarget());

    // Mid-prefill with the image still running: no raise, and no shrink either.
    try std.testing.expect(!arb.pollLlmGrowth());
    try std.testing.expectEqual(@as(?u64, 6 << 30), llm.cp.budgetTarget());

    // The image model goes away mid-prefill, edge-less.
    arb.diffusion = null;
    arb.diff_active = false;
    try std.testing.expect(arb.pollLlmGrowth());
    try std.testing.expectEqual(@as(?u64, 22 << 30), llm.cp.budgetTarget());
    // Published, not applied: the worker owns its own device context.
    try std.testing.expectEqual(@as(?u64, null), llm.applied);
    llm.participant().pollAndApply();
    try std.testing.expectEqual(@as(?u64, 22 << 30), llm.applied);
}

test "Arbiter.pollLlmGrowth: never shrinks a running turn, and no-ops when idle" {
    // Growth only. Mid-forward `usage` carries the pass's transient buffers, so a
    // plan taken from here reads the LLM as fatter than it is; acting on that would
    // migrate layers to the HOST in the middle of the prefill this speeds up.
    var llm: MockModel = .{ .used = 20 << 30, .want = 20 << 30, .floor_b = 2 << 30, .is_busy = true };
    var diff: MockModel = .{ .used = 2 << 30, .floor_b = 0 };
    var arb: Arbiter = .{
        .llm = llm.participant(),
        .diffusion = diff.participant(),
        .limit = 22 << 30,
        .llm_share = 6 << 30,
    };
    arb.rebalance();
    const settled = llm.cp.budgetTarget();

    // The peer grows (a queued image loading): the plan now wants the LLM smaller.
    diff.used = 10 << 30;
    diff.want = 10 << 30;
    try std.testing.expect(!arb.pollLlmGrowth());
    try std.testing.expectEqual(settled, llm.cp.budgetTarget()); // untouched
    // Shrinking still happens, on the edge that owns it.
    arb.setDiffusionActive(true);
    try std.testing.expectEqual(@as(?u64, 6 << 30), llm.cp.budgetTarget());

    // Idle LLM: `rebalance` owns the steady state, so the poll declines outright
    // (it would otherwise apply on the CALLING thread, which is the whole reason
    // settle distinguishes the two).
    llm.is_busy = false;
    diff.used = 0;
    diff.want = 0;
    try std.testing.expect(!arb.pollLlmGrowth());
}

test "Arbiter.diffusionBudget: uninitialized budgets mean auto (0), not the 256 MiB floor" {
    // Regression: pure image-studio mode (no LLM session) never runs the meter
    // policy, so `limit` stays 0. Returning max(256 MiB, 0) there is a hard 256 MiB
    // pin budget that pins a sliver of the image model and evicts the rest. 0 is the
    // pipeline's AUTO sentinel.
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
