//! The engine driver: everything tp-gui's app loop did with the LLM session,
//! the diffusion engine and the VRAM arbiter, as one struct so a probe can
//! script it and tp-serve can host it. Nothing here names a widget.
//!
//! Threads. The owner's thread is the engine thread: it calls the `maybe*`
//! hooks once per iteration and it alone dereferences `session` while `loading`
//! is false. The loader thread builds a session and publishes it under
//! `session_mu`; the diffusion worker reaches the session only through the
//! coordinator hooks, under the same mutex; the LLM worker reaches the image
//! model only through `llmForeignReclaim`, under `diff_mu`.
//!
//! Lock order: `session_mu` ABOVE `diff_mu`, `carry_mu` innermost. Each worker
//! thread takes exactly one: the diffusion worker takes `session_mu` to reach
//! the LLM, the LLM worker takes `diff_mu` to reach diffusion. The engine
//! thread nests them in that order and only there, via `applyMeterPolicy` ->
//! `Arbiter.rebalance` -> an idle LLM's settle -> `residency.promoteBack`, which
//! asks the image model for card space when its own promote won't fit. That
//! nesting cannot contend with the LLM worker's own `llmForeignReclaim`: `settle`
//! applies directly only while the LLM is idle, and the worker path only runs
//! while it is busy. `Diffuser.res_mu` sits below both and is only ever tryLock'd
//! from a foreign thread, so it cannot participate in a cycle either.
//!
//! One Driver per process: `tp.safetensors.read_mode` and the NVML handle are
//! process-wide, and the session's `replan` callback carries no context.
const std = @import("std");
const tp = @import("TensorPencil");
const vram = tp.vram;
const chat = @import("chat.zig");
const diffuser = @import("diffuser.zig");
const sysmon = @import("sysmon.zig");
const config = @import("shared").config;
const pipeline_map = @import("shared").pipeline_map;
const model_spec = @import("shared").model_spec;

const log = std.log;

/// An image attached before the LLM has loaded (the lazy first message): its
/// decoded RGB, fed to `attachImage` once a session exists, plus a display RGBA
/// for the pre-load thumbnail strip.
pub const StagedImage = struct { rgb: []u8, rgba: []u8, width: usize, height: usize };

/// In-flight LLM state saved on an unload-while-paused: the raw `ids` (prompt +
/// partial open response) carried across the unload so a reload can reprefill
/// and continue that exact response. `carry` holds the display transcript.
pub const LlmSuspend = struct { ids: []u32, midturn: bool };

/// Cancel and pause, set from any thread and applied ahead of every queued
/// request on the engine thread's next pass, so neither waits behind an adopt
/// or a settings change. Applied THERE rather than on the caller's thread: the
/// session and the image engine are the engine thread's to free, and a
/// foreign thread holding either pointer is the bug that thread exists to
/// prevent.
pub const Urgent = struct {
    llm_cancel: std.atomic.Value(bool) = .init(false),
    img_cancel_all: std.atomic.Value(bool) = .init(false),
    /// Image ids to cancel, 0 = free slot. A producer claims a free slot by
    /// cmpxchg; when none is free the cancel goes through the inbox instead.
    img_cancel: [img_cancel_slots]std.atomic.Value(u64) = [_]std.atomic.Value(u64){.init(0)} ** img_cancel_slots,
    /// 0 unchanged, 1 pause, 2 resume.
    llm_pause: std.atomic.Value(u8) = .init(0),
    img_pause: std.atomic.Value(u8) = .init(0),
    /// Inbox items accepted so far (`Host.post` bumps it), and the count when
    /// the most recent urgent verb arrived. Skipping the queue must mean
    /// "ahead of what was queued AFTER me", not "ahead of everything": a stop
    /// that followed a message in would otherwise cancel nothing and the
    /// message would be sent anyway. The wait is one drain, since the engine
    /// pass applies these again once the queue is empty.
    posted: std.atomic.Value(u64) = .init(0),
    urgent_at: std.atomic.Value(u64) = .init(0),

    pub const img_cancel_slots = 8;

    /// Mark an urgent verb as arriving behind everything queued up to now.
    pub fn stamp(self: *Urgent) void {
        self.urgent_at.store(self.posted.load(.acquire), .release);
    }

    /// Has the queue reached the point the last urgent verb arrived at.
    pub fn due(self: *const Urgent, handled: u64) bool {
        return self.urgent_at.load(.acquire) <= handled;
    }

    /// Queue an image cancel (any thread). False when every slot is taken.
    pub fn pushImgCancel(self: *Urgent, id: u64) bool {
        if (id == 0) return true;
        for (&self.img_cancel) |*slot| if (slot.load(.acquire) == id) return true;
        for (&self.img_cancel) |*slot| {
            if (slot.cmpxchgStrong(0, id, .acq_rel, .acquire) == null) return true;
        }
        return false;
    }

    /// The next queued image cancel, freeing its slot; null when none is queued.
    pub fn takeImgCancel(self: *Urgent) ?u64 {
        for (&self.img_cancel) |*slot| {
            const id = slot.swap(0, .acq_rel);
            if (id != 0) return id;
        }
        return null;
    }
};

test "the urgent image-cancel ring holds eight ids, drops duplicates and refuses a ninth" {
    var u: Urgent = .{};
    for (1..9) |i| try std.testing.expect(u.pushImgCancel(i));
    try std.testing.expect(u.pushImgCancel(3)); // already queued
    try std.testing.expect(!u.pushImgCancel(9)); // full: the caller takes the slow path
    try std.testing.expect(u.pushImgCancel(0)); // "none" needs no slot
    var seen: u64 = 0;
    while (u.takeImgCancel()) |id| seen |= @as(u64, 1) << @intCast(id);
    try std.testing.expectEqual(@as(u64, 0x1FE), seen);
    try std.testing.expectEqual(@as(?u64, null), u.takeImgCancel());
    try std.testing.expect(u.pushImgCancel(9));
}

test "an urgent verb jumps the queue behind it, never the queue ahead of it" {
    var u: Urgent = .{};
    // Nothing queued: due at once.
    u.stamp();
    try std.testing.expect(u.due(0));
    // A message queued, then a stop: the stop must not be applied until the
    // message it followed has been handled, or it cancels nothing and the
    // message is sent anyway.
    _ = u.posted.fetchAdd(1, .release); // chat_submit into the inbox
    u.stamp(); // chat_cancel behind it
    try std.testing.expect(!u.due(0));
    try std.testing.expect(u.due(1));
    // And a verb that arrives with the queue already drained waits for nothing.
    u.stamp();
    try std.testing.expect(u.due(1));
}

/// A `newChat` or `adoptTranscript` that arrived while a turn was in flight or
/// a load was running, applied once the transcript is this thread's again.
pub const PendingTranscript = union(enum) { new, adopt: std.ArrayList(chat.Message) };

var g_instance: ?*Driver = null;

/// How long a worker waits for the engine thread to enact a release. Generous
/// against the work involved (freeing a multi-GiB pipeline, behind at most one
/// frame, since `requestRelease` wakes the loop) and it is not a poll interval:
/// the wait ends the moment the release lands, and the only way to reach the
/// deadline is an engine thread that has stopped servicing frames, in which case
/// failing the allocation is the right answer.
const release_wait_ns: i96 = 2 * std.time.ns_per_s;

/// Interval between mid-turn re-plans. A prefill chunk is tens of milliseconds
/// on the GPU, so this is per-chunk work; a quarter second is far more often
/// than a peer's residency can meaningfully move, and the poll itself is a plan
/// over counters, no device query.
const replan_interval_ns: i96 = 250 * std.time.ns_per_ms;

pub const Driver = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    /// Repaints the owner (pushed from worker threads too).
    wake: *const fn () void,
    /// The settings in force: what the engines were last reconciled against.
    /// `applySettings` diffs a new config against this to decide what a change
    /// costs, then adopts it.
    settings: config.Config,
    /// A copy taken when a load is armed, so the loader reads settings nobody
    /// edits under it.
    load_settings: config.Config = .{},
    /// The one thread that touches the session, the image queue and the
    /// arbiter; `assertEngineThread` names it at every such entry.
    engine_thread: std.Thread.Id,
    urgent: Urgent = .{},

    // The session (LLM + optional vision) is (re)loaded on a background thread
    // whenever settings change, so it can be swapped without a restart. Only read
    // by the engine thread while `loading` is false (release/acquire hand-off
    // from the loader), so the pointer swap is race-free without a lock.
    session: ?*chat.Session = null,
    /// The single owner of LLM<->diffusion VRAM arbitration. Its `llm` participant
    /// is (re)bound to `session` under `session_mu` on every load/unload.
    arbiter: vram.Arbiter = .{},
    session_arena: ?*std.heap.ArenaAllocator = null, // load-once weights, freed on reload
    loading: std.atomic.Value(bool) = .init(false),
    loader: ?std.Thread = null,
    reload_requested: bool = false,
    /// Text stashed when the first message is sent with no LLM resident: the
    /// model lazy-loads, then this is auto-submitted (see `maybeStartReload`).
    pending_submit: ?[]u8 = null,
    staged: std.ArrayList(StagedImage) = .empty,

    // Cached "does the configured LLM support a reasoning block?" answer, so the
    // thinking toggle can show before the model loads. Re-probed only when the
    // configured model path changes.
    think_probe_path: [config.max_path]u8 = undefined,
    think_probe_len: usize = 0,
    think_probe_result: bool = false,
    think_probe_effort: bool = false,
    think_probe_valid: bool = false,
    /// Same memo for the weight-noise capability probe. Separate because the two
    /// answers come from different things (a chat template vs. which kernels were
    /// wired) and a model can support either without the other.
    noise_probe_path: [config.max_path]u8 = undefined,
    noise_probe_len: usize = 0,
    noise_probe_result: bool = false,
    noise_probe_valid: bool = false,

    /// The diffusion engine, persistent across chat<->studio switches. Owns the
    /// unified image queue/history. Built when a diffusion model is configured;
    /// its pipeline still loads lazily on the first image.
    diffuser: ?diffuser.Diffuser = null,
    /// VRAM meter handle positions (fractions of the card): split = LLM|diffusion
    /// contention boundary, limit = ceiling. The owner's meter mutates them in
    /// place on drag and calls `applyMeterPolicy` on release.
    split: f32 = 0.60,
    limit: f32 = 0.95,
    /// The card's size in bytes, from whoever could measure it before anything
    /// was loaded (`host.zig` probes the device-local heap at start). On a host
    /// with no LLM there is no live context to ask until the first image is
    /// already loading, and that image is the one that has to be bounded.
    card_bytes_hint: u64 = 0,
    /// The diffusion-only allowance last handed to the arbiter, so re-running
    /// the policy on an idle loop is free and silent.
    diff_only_limit: u64 = 0,
    /// Eject state: set when the user asks to unload a model. If the model is busy
    /// the request stays ARMED and fires once it (and the shared image queue) go
    /// idle, see `maybeProcessEjects`.
    llm_eject_armed: bool = false,
    diff_eject_armed: bool = false,
    /// LLM pause, mirrored here so it survives an unload (the gate itself lives on
    /// the Session, which dies on unload, unlike the diffuser's gate).
    llm_paused: bool = false,
    llm_suspend: ?LlmSuspend = null,
    /// Set when a reload should CONTINUE a suspended mid-turn response.
    pending_continue: bool = false,
    /// Set when regenerate is asked for while the LLM is unloaded: lazy-load, then
    /// regenerate the last reply once the carried transcript is adopted.
    pending_regenerate: bool = false,
    pending_transcript: ?PendingTranscript = null,

    /// Guards `session` teardown against the diffusion WORKER thread, which reads
    /// the session in its VRAM-coordinator hooks (budget/reclaim).
    session_mu: std.Io.Mutex = std.Io.Mutex.init,
    /// The mirror: guards `diffuser` / `arbiter.diffusion` against the LLM WORKER
    /// thread, which reaches the image model through `llmForeignReclaim`.
    diff_mu: std.Io.Mutex = std.Io.Mutex.init,

    /// The transcript carried across a model-swap reload (detached from the old
    /// session before teardown, adopted by the new one) so a settings save never
    /// wipes the chat. Owned by `gpa`; freed if there is no new session to adopt.
    carry: ?std.ArrayList(chat.Message) = null,
    /// Outcome notes for the model that arrived with no session to hold them:
    /// an eject or a reload window. Handed over as soon as one is live, so a
    /// render that finished meanwhile is still reported. Engine thread only.
    pending_notes: std.ArrayList([]u8) = .empty,
    /// Guards `carry`, which the `!loading` gate CANNOT protect the way it protects
    /// `session`: the owner reads the carried transcript precisely while a load
    /// runs. Held across the WALK, not just the fetch, which makes it the
    /// INNERMOST lock: the two places that nest (the detach paths) take it under
    /// `session_mu`, never the other way.
    carry_mu: std.Io.Mutex = std.Io.Mutex.init,

    load_err: ?anyerror = null,
    /// Tracks the LLM's busy edge for `maybeRefreshMeterPolicy`.
    llm_was_busy: bool = false,
    /// High-water mark of OUR unattributed card footprint (CUDA context, JIT'd
    /// modules, library workspaces, UI textures). See `applyMeterPolicy`.
    untracked_high_water: u64 = 0,
    last_retry_ns: i96 = 0,
    /// Written only under `diff_mu`.
    last_replan_ns: i96 = 0,
    /// The configured checkpoint's family, from the FILE (see `configuredFamily`).
    family_cache: ?model_spec.Cache = null,
    /// A measured diffusion peak that grew past what `settings` recorded. The
    /// owner takes it (`takePeakUpdate`) and persists it; the engine keeps no disk.
    peak_update: ?u64 = null,

    /// In-place, because worker callbacks capture `self`. Panics on a second
    /// instance in the process.
    pub fn init(self: *Driver, gpa: std.mem.Allocator, io: std.Io, wake: *const fn () void, settings: *const config.Config) void {
        if (g_instance != null) @panic("one Driver per process");
        self.* = .{
            .gpa = gpa,
            .io = io,
            .wake = wake,
            .settings = settings.*,
            .engine_thread = std.Thread.getCurrentId(),
        };
        g_instance = self;
        self.adoptMeterFractions(settings);
        applyWeightRead(settings.weight_read);
    }

    /// Take the meter handles off the settings. They ride there as well as on
    /// the `meter` verb, and a driver built straight from a config (a probe, a
    /// headless host) never sees the verb: without this it runs the whole
    /// session on the defaults, and on a small card the reserve is the
    /// difference between streaming and a GPU reset. The verb still overrides
    /// live, which is what a drag on the bar is.
    fn adoptMeterFractions(self: *Driver, cfg: *const config.Config) void {
        self.split = std.math.clamp(cfg.vram_split, 0.02, 0.96);
        self.limit = std.math.clamp(cfg.vram_limit_frac, 0.10, 0.985);
    }

    /// Tear down at exit, in dependency order: the loader is joined FIRST by the
    /// owner (`joinLoader`), which then flushes whatever it persists, then this
    /// frees the image engine (joins its worker) before the session any of its
    /// threads could touch.
    pub fn deinit(self: *Driver) void {
        self.joinLoader();
        self.freeDiffuser();
        if (self.session) |s| {
            s.be.bindThread();
            s.deinit();
        }
        if (self.session_arena) |a| {
            a.deinit();
            self.gpa.destroy(a);
        }
        self.freeCarry();
        self.dropNotes();
        self.pending_notes.deinit(self.gpa);
        self.freeLlmSuspend();
        self.dropPendingTranscript();
        if (self.pending_submit) |p| self.gpa.free(p);
        self.clearStaged();
        self.staged.deinit(self.gpa);
        if (self.family_cache) |*c| c.deinit();
        if (g_instance == self) g_instance = null;
    }

    pub fn joinLoader(self: *Driver) void {
        if (self.loader) |t| t.join();
        self.loader = null;
    }

    /// The session, when this thread may touch it: null while the loader owns it.
    pub fn uiSession(self: *Driver) ?*chat.Session {
        self.assertEngineThread();
        if (self.loading.load(.acquire)) return null;
        return self.session;
    }

    pub fn assertEngineThread(self: *const Driver) void {
        std.debug.assert(std.Thread.getCurrentId() == self.engine_thread);
    }

    /// Apply the flags in `urgent`. Runs either side of the inbox drain;
    /// `handled` is how many queued requests this host has taken so far.
    pub fn applyUrgent(self: *Driver, handled: u64) void {
        self.assertEngineThread();
        const u = &self.urgent;
        if (!u.due(handled)) return;
        if (u.llm_cancel.swap(false, .acq_rel)) self.cancelLlm();
        if (u.img_cancel_all.swap(false, .acq_rel)) self.cancelAllImages();
        while (u.takeImgCancel()) |id| self.cancelImage(id);
        switch (u.llm_pause.swap(0, .acq_rel)) {
            1 => self.setLlmPaused(true),
            2 => self.setLlmPaused(false),
            else => {},
        }
        switch (u.img_pause.swap(0, .acq_rel)) {
            1 => self.setDiffPaused(true),
            2 => self.setDiffPaused(false),
            else => {},
        }
    }

    // The cancel and pause verbs, whether they arrived urgently or through the
    // inbox. Engine thread.

    pub fn cancelLlm(self: *Driver) void {
        if (self.uiSession()) |s| return s.requestCancel();
        // Nothing resident to cancel: the stop is for the message (or the
        // regenerate) waiting on the load, which `maybeStartReload` would
        // otherwise submit the moment the load finished.
        if (self.pending_submit) |p| self.gpa.free(p);
        self.pending_submit = null;
        self.pending_regenerate = false;
    }

    pub fn cancelImage(self: *Driver, id: u64) void {
        self.assertEngineThread();
        const d = &(self.diffuser orelse return);
        const gi = d.byId(id) orelse return;
        gi.cancel.store(true, .release);
        gi.wake();
        // A worker parked at the pause gate would never see the flag.
        d.wakePaused();
    }

    /// Forget an image the client has taken delivery of. True when one went.
    pub fn dropImage(self: *Driver, id: diffuser.ImageId) bool {
        self.assertEngineThread();
        const d = &(self.diffuser orelse return false);
        return d.drop(id);
    }

    pub fn cancelAllImages(self: *Driver) void {
        self.assertEngineThread();
        if (self.diffuser) |*d| d.cancelAll();
    }

    pub fn setLlmPaused(self: *Driver, paused: bool) void {
        if (paused != self.llm_paused) self.toggleLlmPause();
    }

    pub fn setDiffPaused(self: *Driver, paused: bool) void {
        if (paused != self.diffPaused()) self.toggleDiffPause();
    }

    // ── VRAM meter policy ─────────────────────────────────────────────────────

    /// Resolve the meter handles against the live card and hand the result to the
    /// arbiter, which drives BOTH models (soft residency): the limit is a WHOLE-CARD
    /// ceiling, so our budget for LLM + diffusion is `limit − system` (system =
    /// OS/desktop + CUDA-context overhead we don't control), and the split handle is
    /// the LLM's guaranteed share under contention.
    ///
    /// Carrying the diffusion half by hand does not work: settling the LLM first and then
    /// offering diffusion `available - (the LLM's just-shrunk usage)` gives a second target
    /// that algebraically cancels to diffusion's own residency, so the image model never
    /// yields a byte. Both targets come from one `Arbiter.plan`, which is also what makes
    /// the split handle mean something. No-op with no session or mid-load.
    pub fn applyMeterPolicy(self: *Driver) void {
        self.assertEngineThread();
        if (self.loading.load(.acquire)) return;
        const s = self.session orelse return self.applyDiffusionOnlyPolicy();
        // cuMemGetInfo reads the CALLING thread's current CUDA context, and this
        // runs on the engine thread, which starts with none bound (the LLM's
        // context is created on the loader thread). Bind it first, or the query
        // fails and returns zeros, and that silent zero skips `setBudgets`
        // entirely, leaving the arbiter uninitialized and the first message
        // mass-offloading.
        s.be.bindThread();
        const mi = s.be.ctx.memGetInfo() orelse {
            log.warn("[vram] meter policy skipped: VRAM query failed{s} — arbiter budgets NOT updated", .{
                if (s.be.ctx.isLost()) " (the CUDA context has been lost; restart to use the GPU)" else "",
            });
            return;
        };
        const total: u64 = mi.total;
        // ONE coherent pass over the card, then one rule (see `vram.resolve`).
        //
        // This replaced `budget = limit - system`, where `system` was the residual
        // `device_used - our_tracked`. That put an unreliable number alone on the
        // right-hand side: it is sampled here, right after a load, when our own CUDA
        // context / JIT'd modules / library workspaces do not exist yet, so it read
        // ~1.1 GiB low, the budget came out that much too generous, the LLM promoted
        // every layer to fill it, and the next batched allocation OOM'd and offloaded
        // them straight back. `resolve` frames the handle as a RESERVE instead, where
        // an under-read can only fall back to what the user asked for.
        const tf: f32 = @floatFromInt(total);
        const llm_res: u64 = s.be.deviceUsed();
        const diff_res: u64 = if (self.diffuser) |*d| d.vramBytes() else 0;
        var card: vram.Card = .{
            .total = total,
            .foreign = null, // no NVML: degrade to the residual (conservative)
            .ours_tracked = llm_res + diff_res,
            .device_used = total -| mi.free,
        };
        if (sysmon.nvml()) |nv| {
            if (nv.selfUsed()) |proc| {
                // Same pass as `mi` above. `foreign` is then OTHER PROCESSES ONLY, and
                // our unattributed bytes land in `ours_total` where they belong.
                card.foreign = card.device_used -| proc;
                card.ours_total = proc;
            }
        }
        // High-water the untracked term: it is ~0 on a cold model and grows as modules
        // and workspaces are JIT'd/allocated. Letting it fall back would re-inflate the
        // budget mid-session and restart the promote -> OOM -> offload cycle. It is
        // deliberately NOT reset per model load, the CUDA context and compiled modules
        // behind most of it outlive any one session.
        const res = vram.resolve(.{ .fraction = self.limit }, card, self.untracked_high_water);
        self.untracked_high_water = res.untracked;

        const available: u64 = res.tracked; // cap on LLM + diffusion TRACKED bytes
        // The split handle stays a fraction of the CARD (that is what the meter draws),
        // clamped into what is actually ours to give away.
        const share: u64 = @min(@as(u64, @intFromFloat(self.split * tf)), available);

        // Guarded so a direct (idle) settle can't race the diffusion worker's reclaim
        // hook, both touch the LLM context. The diffusion half of the rebalance takes
        // the engine's own `res_mu` internally.
        self.session_mu.lockUncancelable(self.io);
        defer self.session_mu.unlock(self.io);
        s.vram_limit = available;
        s.vram_share = share;
        s.vram_budget = available;
        var rbuf: [200]u8 = undefined;
        log.info("[vram] card {d} MiB · limit {d} · {s}{s}", .{
            total >> 20,
            @as(u64, @intFromFloat(self.limit * tf)) >> 20,
            res.render(&rbuf),
            if (card.foreign == null) " (no NVML: foreign is a residual)" else "",
        });
        vram.logResidency("LLM", s.residencyNeed(), s.residencyHave());
        // The itemized counterpart: `logResidency` reports `scratch` as a subtraction,
        // this reports what each allocation path actually claimed. When the two
        // disagree the difference is the thing to chase.
        s.logMemTags(card.ours_total);
        self.arbiter.setBudgets(available, share);
        // The budgets in force are now the LLM-era ones, so the diffusion-only
        // memo no longer describes them: without this, ejecting the LLM finds
        // the same figure it last recorded, returns early, and leaves diffusion
        // sharing a card nothing else is on.
        self.diff_only_limit = 0;
    }

    /// The meter policy on a host with nothing but an image engine.
    ///
    /// The SPLIT is meaningless with nothing to split against, so diffusion is
    /// handed the whole allowance. The RESERVE is not, and it binds here
    /// exactly as it does beside an LLM: it is the only thing standing between
    /// the weight cache and the whole card.
    ///
    /// Without this the arbiter is never given a limit, `diffusionBudget`
    /// answers 0, and the pipeline pins against live free VRAM with nothing
    /// held back for the per-image working set. Measured on a 4 GB Arc: a
    /// 3645 MiB peak against 3196 MiB free, a step that took 444 s, and the
    /// kernel resetting the GPU.
    ///
    /// The card total comes from the start-up probe in preference to the live
    /// pipeline: it is a plain number needing no context, where `memGetInfo`
    /// reads the CALLING thread's context and this is not the thread that owns
    /// the diffusion one.
    fn applyDiffusionOnlyPolicy(self: *Driver) void {
        // It is named for the case it is for. With an LLM resident the budgets
        // are `applyMeterPolicy`'s, which subtract what the card is carrying;
        // overwriting them with a raw fraction of the whole card is the
        // promote -> OOM -> offload cycle again.
        if (self.session != null) return self.applyMeterPolicy();
        const d = &(self.diffuser orelse return);
        const total: u64 = if (self.card_bytes_hint != 0)
            self.card_bytes_hint
        else if (d.vramInfo()) |vi| vi.total else 0;
        if (total == 0) return; // nothing known about the card; leave it alone
        const available: u64 = @intFromFloat(self.limit * @as(f32, @floatFromInt(total)));
        if (available == 0 or available == self.diff_only_limit) return;
        self.diff_only_limit = available;
        log.info("[vram] card {d} MiB · no LLM: reserve leaves {d} MiB to diffusion (split ignored)", .{
            total >> 20, available >> 20,
        });
        self.arbiter.setBudgets(available, 0);
    }

    /// Main-loop hook: re-resolve the meter policy when the LLM finishes a turn.
    ///
    /// The arbiter's residency targets come from `residency.demand`, and for an
    /// LLM-only session the plan behind them was computed exactly once, at load,
    /// when the model is COLD. Its RoPE tables, activation/logits scratch and dequant
    /// buffers are not allocated until the first forward, so `demand`'s cold bound
    /// cannot see them and reports less than full residency costs. That
    /// under-estimate then sticks as the ceiling for the whole session: the LLM
    /// offloads layers to obey a target below its real footprint while GiBs of the
    /// card sit unused.
    ///
    /// Re-planning at a turn boundary keeps residency tracking a budget that moves:
    /// `system` steps once as our untracked CUDA/library footprint materializes, and
    /// the KV grows all session.
    ///
    /// BOTH edges matter, because `Arbiter.plan` gates the split handle on who is
    /// actually working:
    ///   idle -> busy   the LLM starts a turn, so the split now binds and it reclaims
    ///                  the layers an image model borrowed while it was idle.
    ///   busy -> idle   the turn ended, so a working diffuser may take what it needs.
    /// Edge-triggered, so an idle loop re-plans nothing; cheap when it does fire (one
    /// memGetInfo plus a plan).
    pub fn maybeRefreshMeterPolicy(self: *Driver) void {
        self.assertEngineThread();
        // Same rule as the renderers: the loader thread owns the session while a load
        // is in flight, so `busy()` below would read freed memory. Nothing is missed:
        // `maybeStartReload` re-applies the policy the moment the fresh session is
        // published, and clearing the edge here makes its first turn a real idle->busy.
        if (self.loading.load(.acquire)) {
            self.llm_was_busy = false;
            return;
        }
        const s = self.session orelse {
            self.llm_was_busy = false;
            // No LLM means no busy edge to wait for, but the reserve still has
            // to reach the arbiter, and a dial the user moves has to land.
            self.applyDiffusionOnlyPolicy();
            return;
        };
        const busy = s.busy();
        defer self.llm_was_busy = busy;
        if (self.llm_was_busy != busy) return self.applyMeterPolicy();
        self.retryUnsettledPlan();
    }

    /// Main-loop hook: re-run a plan a model did not actually enact.
    ///
    /// `Diffuser.giveUpToBudget` tryLocks its residency mutex and declines rather
    /// than block a foreign thread, so a yield can be lost to a frame in which the
    /// pump happened to hold it. Nothing noticed: the settle was fire-and-forget and
    /// the next rebalance waits on an external edge that may never come, leaving the
    /// LLM squeezed next to an image model that agreed to shrink and didn't. The
    /// arbiter bounds its own retries, so this cannot spin on a peer that genuinely
    /// has nothing left; the throttle only keeps a lost race off the frame path.
    fn retryUnsettledPlan(self: *Driver) void {
        if (!self.arbiter.unsettled) return;
        const now = std.Io.Clock.real.now(self.io).nanoseconds;
        if (now - self.last_retry_ns < 250 * std.time.ns_per_ms) return;
        self.last_retry_ns = now;
        self.session_mu.lockUncancelable(self.io);
        defer self.session_mu.unlock(self.io);
        _ = self.arbiter.retryUnsettled();
    }

    // ── Pause and eject ───────────────────────────────────────────────────────

    /// Diffusion's gate lives on the persistent Diffuser, so it IS the source of
    /// truth (queried live, survives an unload). The LLM mirrors its state in
    /// `llm_paused` because the gate dies with the session on unload.
    pub fn diffPaused(self: *Driver) bool {
        return if (self.diffuser) |*d| d.isPaused() else false;
    }

    pub fn toggleLlmPause(self: *Driver) void {
        self.assertEngineThread();
        const now_paused = !self.llm_paused;
        self.llm_paused = now_paused;
        if (self.loading.load(.acquire)) return; // applied to the fresh gate on publish
        if (self.session) |s| {
            // Resident: drive the session gate (unpause also dispatches a turn that
            // was queued while paused, see Session.setPaused).
            s.setPaused(now_paused);
        } else if (!now_paused) {
            // Resuming with nothing resident: fire any load we HELD while paused, a
            // message / regenerate stashed by `submit` / `requestRegenLoad` (both set
            // reload_requested), or a suspended turn to resume. Wake a frame so
            // maybeStartReload runs now that the gate is lifted. Nothing pending ⇒
            // nothing loads (a bare pause->resume on a cold engine is a no-op).
            if (self.reload_requested or self.pending_submit != null or self.pending_regenerate or self.llm_suspend != null) {
                self.reload_requested = true;
                self.wake();
            }
        }
    }

    pub fn toggleDiffPause(self: *Driver) void {
        self.assertEngineThread();
        if (self.diffuser) |*d| {
            d.setPaused(!d.isPaused());
            // Wake a frame so the next pump() runs at once: on resume it loads +
            // starts any image queued while paused (pump defers all loads while
            // paused, so nothing loaded until now). Harmless on pause.
            self.wake();
        }
    }

    /// Main-loop hook: carry out any armed eject once ITS OWN model is idle. Each
    /// model ejects independently, the LLM can drop while diffusion is still
    /// generating (it isn't generating anything, so there's nothing to wait for),
    /// and vice versa. A model that's busy when clicked stays armed and ejects the
    /// moment it finishes.
    pub fn maybeProcessEjects(self: *Driver) void {
        self.assertEngineThread();
        if (self.diff_eject_armed) {
            if (self.diffuser) |*d| {
                if (d.isPaused()) {
                    // Unload-while-paused: snapshot the in-flight image to host, then
                    // free the weights and KEEP the queue (incl. the suspended image),
                    // which resumes on unpause. Free even with pending images, they're
                    // parked by the pause gate anyway.
                    if (d.busyNow()) {
                        d.requestSuspend(); // worker snapshots + exits; poll next frame
                    } else {
                        d.reapAndFree();
                        self.diff_eject_armed = false;
                        self.applyMeterPolicy();
                    }
                } else if (!d.busyNow() and !d.hasPending()) {
                    d.freeSession();
                    self.diff_eject_armed = false;
                    // Diffusion is gone: let the LLM borrow the freed VRAM back.
                    self.applyMeterPolicy();
                }
            } else self.diff_eject_armed = false;
        }
        if (self.llm_eject_armed) {
            if (self.session == null or self.loading.load(.acquire)) {
                self.llm_eject_armed = false; // nothing loaded / a (re)load is in flight
            } else if (self.session) |s| {
                if (self.llm_paused and s.busy()) {
                    // Unload-while-paused: the worker is parked mid-decode. Ask it to
                    // suspend (stop with the turn left OPEN); we finish the unload once
                    // it clears below.
                    s.requestSuspend();
                } else if (!s.busy()) {
                    // Fire as soon as the LLM itself is idle, do NOT wait on
                    // diffusion. The worker only touches the session through the
                    // coordinator hooks, which unloadLlm serializes with session_mu.
                    // If we suspended a mid-turn response, carry its raw `ids` so the
                    // reload can reprefill + continue it.
                    if (self.llm_paused and s.suspended_midturn) self.saveLlmSuspend(s);
                    self.unloadLlm();
                    self.llm_eject_armed = false;
                }
            }
        }
    }

    /// Carry the suspended LLM's raw `ids` (prompt + partial open response) across an
    /// unload-while-paused so a reload can reprefill + continue it. `carry` holds
    /// the display transcript alongside. Drains pending bytes first so the displayed
    /// text matches the tokens.
    fn saveLlmSuspend(self: *Driver, s: *chat.Session) void {
        s.poll(); // drain streamed bytes into messages so display matches `ids`
        const ids = self.gpa.dupe(u32, s.ids.items) catch |err| {
            log.err("save suspend ids: {t}", .{err});
            return;
        };
        if (self.llm_suspend) |old| self.gpa.free(old.ids); // one suspend at a time
        self.llm_suspend = .{ .ids = ids, .midturn = s.suspended_midturn };
    }

    pub fn freeLlmSuspend(self: *Driver) void {
        if (self.llm_suspend) |sus| {
            self.gpa.free(sus.ids);
            self.llm_suspend = null;
        }
        self.pending_continue = false;
    }

    /// Fully unload the LLM (free its VRAM) while KEEPING the conversation: the
    /// transcript is detached into `carry` (rendered read-only until a message
    /// reloads + replays it), never wiped. Runs synchronously on the engine thread
    /// (LLM is idle here); the teardown is serialized with the diffusion worker's
    /// session access via `session_mu`.
    pub fn unloadLlm(self: *Driver) void {
        const s = self.session orelse return;
        if (s.worker) |t| { // idle here, but be safe
            t.join();
            s.worker = null;
        }
        self.session_mu.lockUncancelable(self.io);
        s.be.bindThread(); // context current on THIS thread to free its device memory
        {
            self.carry_mu.lockUncancelable(self.io);
            defer self.carry_mu.unlock(self.io);
            self.carry = s.detachTranscript();
        }
        s.deinit();
        self.session = null;
        self.arbiter.llm = null; // participant points into the freed session
        self.session_mu.unlock(self.io);
        if (self.session_arena) |a| {
            a.deinit();
            self.gpa.destroy(a);
            self.session_arena = null;
        }
        // Diffusion (if resident) can now borrow the whole card: with no session,
        // imageBudget returns 0 (pin all free VRAM) on the next image.
        self.wake();
    }

    // ── Attachments and capability probes ─────────────────────────────────────

    /// Can the next message carry an image? True when a vision tower is resident,
    /// or, before the lazy first-message load, when the configured model has one
    /// (an mmproj path is set alongside an LLM). While a load is in flight the
    /// session is off-limits, so only the settings answer.
    pub fn visionAvailable(self: *Driver) bool {
        if (self.uiSession()) |s| return s.visionEnabled();
        return self.settings.vision_tower.opt() != null and self.settings.llm_model.opt() != null;
    }

    /// Attach a decoded RGB image to the next message. Hands it to the live session,
    /// or (lazy first message) stages it and kicks a load. Callers decode the source
    /// and confirm `visionAvailable()` first.
    pub fn attachOrStage(self: *Driver, rgb: []const u8, w: usize, h: usize) void {
        self.assertEngineThread();
        // Only while the session is ours to touch. Mid-load it belongs to the loader
        // thread, and the fall-through is exactly what that case wants anyway:
        // stage the image and let `maybeStartReload` attach it to the fresh session.
        if (self.uiSession()) |s| {
            s.attachImage(rgb, w, h) catch |err| log.err("attach image: {t}", .{err});
            return;
        }
        const rgb_own = self.gpa.dupe(u8, rgb) catch return;
        const rgba = tp.image.rgbToRgba(self.gpa, rgb_own, w, h) catch {
            self.gpa.free(rgb_own);
            return;
        };
        self.staged.append(self.gpa, .{ .rgb = rgb_own, .rgba = rgba, .width = w, .height = h }) catch {
            self.gpa.free(rgb_own);
            self.gpa.free(rgba);
            return;
        };
        // Kick the lazy load so the staged image (and any first message) lands in a
        // session; maybeStartReload drains `staged` once it's live.
        if (!self.loading.load(.acquire)) self.reload_requested = true;
    }

    /// Drop a not-yet-loaded staged attachment by index (pre-session mirror of
    /// `Session.removeAttachment`).
    pub fn removeStaged(self: *Driver, idx: usize) void {
        self.assertEngineThread();
        if (idx >= self.staged.items.len) return;
        const st = self.staged.orderedRemove(idx);
        self.gpa.free(st.rgb);
        self.gpa.free(st.rgba);
    }

    /// Free all staged attachments (load failed, or "new chat" before load).
    pub fn clearStaged(self: *Driver) void {
        for (self.staged.items) |st| {
            self.gpa.free(st.rgb);
            self.gpa.free(st.rgba);
        }
        self.staged.clearRetainingCapacity();
    }

    /// Whether the *configured* LLM (by GGUF architecture) can reason, so the
    /// thinking toggle can show before the model loads. A live session's loaded
    /// family is authoritative; this covers the pre-load window and re-probes
    /// whenever the configured model path changes.
    pub fn configuredSupportsThinking(self: *Driver) bool {
        const path = self.settings.llm_model.opt() orelse {
            self.think_probe_valid = false;
            self.think_probe_len = 0;
            return false;
        };
        if (!(self.think_probe_valid and self.think_probe_len == path.len and
            std.mem.eql(u8, self.think_probe_path[0..self.think_probe_len], path)))
        {
            self.think_probe_result = self.probeThinking(path);
            @memcpy(self.think_probe_path[0..path.len], path);
            self.think_probe_len = path.len;
            self.think_probe_valid = true;
        }
        return self.think_probe_result;
    }

    pub fn configuredSupportsReasoningEffort(self: *Driver) bool {
        _ = self.configuredSupportsThinking();
        return self.think_probe_valid and self.think_probe_effort;
    }

    /// Read the configured GGUF's architecture and map it to reasoning support.
    /// Any failure (missing/unreadable file, unknown arch) -> false.
    fn probeThinking(self: *Driver, path: []const u8) bool {
        self.think_probe_effort = false;
        var gg = tp.Gguf.openHeader(self.gpa, self.io, path) catch return false;
        defer gg.deinit();
        const arch = gg.getStr("general.architecture") orelse return false;
        const fam = tp.llm.chat.familyForArch(arch) orelse return false;
        self.think_probe_effort = tp.llm.chat.familySupportsReasoningEffort(fam);
        return tp.llm.chat.familySupportsThinking(fam);
    }

    /// Whether the weight-noise controls should be offered at all: the loaded model's
    /// answer when there is one, otherwise the CONFIGURED file's.
    ///
    /// The pre-load half is the point. The LLM loads lazily, on the first message,
    /// so gating on a live session hid the controls during exactly the window
    /// someone wants to set them up in.
    pub fn noiseAvailable(self: *Driver) bool {
        if (self.uiSession()) |s| return s.weightNoiseSupported();
        return self.configuredSupportsWeightNoise();
    }

    /// Whether the *configured* checkpoint would honor weight noise, by reading its
    /// header. Re-probed whenever the configured path changes.
    pub fn configuredSupportsWeightNoise(self: *Driver) bool {
        const path = self.settings.llm_model.opt() orelse {
            self.noise_probe_valid = false;
            self.noise_probe_len = 0;
            return false;
        };
        if (!(self.noise_probe_valid and self.noise_probe_len == path.len and
            std.mem.eql(u8, self.noise_probe_path[0..self.noise_probe_len], path)))
        {
            self.noise_probe_result = self.probeWeightNoise(path);
            @memcpy(self.noise_probe_path[0..path.len], path);
            self.noise_probe_len = path.len;
            self.noise_probe_valid = true;
        }
        return self.noise_probe_result;
    }

    fn probeWeightNoise(self: *Driver, path: []const u8) bool {
        var gg = tp.Gguf.openHeader(self.gpa, self.io, path) catch return false;
        defer gg.deinit();
        return tp.llm.session.weightNoiseSupported(&gg);
    }

    // ── Diffusion engine VRAM coordinator ─────────────────────────────────────
    // The engine calls these through a type-erased ctx, which is the Driver.
    // Coordination goes to the resident LLM if there is one, else no-ops
    // (diffusion has the device to itself).

    fn vcEnter(ctx: *anyopaque) void {
        const self: *Driver = @ptrCast(@alignCast(ctx));
        // Image queue started -> the arbiter drives the LLM down to its share. Under
        // the lock: an idle LLM is settled directly on this (engine) thread, which
        // must not race the diffusion worker's reclaim hook (both touch the LLM
        // context). A busy LLM is only published to its control point (an atomic)
        // and yields at its next token.
        self.session_mu.lockUncancelable(self.io);
        defer self.session_mu.unlock(self.io);
        self.arbiter.setDiffusionActive(true);
    }

    fn vcExit(ctx: *anyopaque) void {
        const self: *Driver = @ptrCast(@alignCast(ctx));
        // Queue drained -> idle. The image model's residency becomes opportunistic
        // cache, so the LLM gets priority over it again. One rebalance settles BOTH
        // sides: a `setDiffusionActive` plus a separate hand-written diffusion trim is
        // two passes that disagree.
        //
        // The flag is set directly rather than via `setDiffusionActive` so the ceiling
        // re-resolve below is what triggers the single rebalance; flipping it first
        // would rebalance twice, the first time against a stale `system` reading.
        {
            self.session_mu.lockUncancelable(self.io);
            defer self.session_mu.unlock(self.io);
            self.arbiter.diff_active = false;
        }
        self.recordDiffPeak();
        self.applyMeterPolicy();
    }

    /// Record the diffusion pipeline's measured peak residency when it grows, on
    /// the queue-drain edge (once per batch, not per frame), so the NEXT session's
    /// first image plans against a measurement instead of the file-size bootstrap.
    /// The owner persists it (`takePeakUpdate`).
    fn recordDiffPeak(self: *Driver) void {
        const d = if (self.diffuser) |*x| x else return;
        const peak = d.peakResident();
        if (peak <= self.settings.diff_peak_resident) return;
        const model = self.settings.diffusion_model.opt() orelse return;
        self.settings.diff_peak_resident = peak;
        self.settings.diff_peak_key = config.modelKey(model);
        self.peak_update = peak;
        log.info("[vram] measured diffusion peak {d} MiB (supersedes the checkpoint-size estimate)", .{peak >> 20});
    }

    /// A grown diffusion peak since the last call, with its model key, or null.
    pub fn takePeakUpdate(self: *Driver) ?struct { peak: u64, key: u64 } {
        const p = self.peak_update orelse return null;
        self.peak_update = null;
        return .{ .peak = p, .key = self.settings.diff_peak_key };
    }

    fn vcBudget(ctx: *anyopaque) u64 {
        const self: *Driver = @ptrCast(@alignCast(ctx));
        // Called on the diffusion WORKER thread, serialize with a concurrent LLM
        // eject (unloadLlm) that may be freeing the session right now. Pure read.
        self.session_mu.lockUncancelable(self.io);
        defer self.session_mu.unlock(self.io);
        return self.arbiter.diffusionBudget();
    }

    fn vcReclaim(ctx: *anyopaque, needed: u64) u64 {
        const self: *Driver = @ptrCast(@alignCast(ctx));
        // Worker thread; the reclaim hook binds the LLM context, so it must not race
        // an eject freeing that context. (LLM idle is checked inside imageReclaim.)
        self.session_mu.lockUncancelable(self.io);
        defer self.session_mu.unlock(self.io);
        return if (self.session) |s| s.imageReclaim(needed) else 0;
    }

    fn coordinator(self: *Driver) diffuser.VramCoordinator {
        return .{ .ctx = @ptrCast(self), .enter = vcEnter, .exit = vcExit, .budget = vcBudget, .reclaim = vcReclaim };
    }

    // ── The reverse direction: LLM reclaims from diffusion ────────────────────
    // `vcReclaim` lets a diffusion worker migrate LLM layers to the host mid-image.
    // This is its mirror: an LLM allocation that does not fit under a
    // resident-but-idle image model reaches that model's bytes (separate device
    // contexts, and LLM weights are all pinned so its own eviction ladder reclaims
    // nothing).
    //
    // Two callers, both of which must be able to reach a peer in another context:
    // the last rung of `cuda.Backend`'s OOM ladder (LLM worker thread), and
    // `residency.promoteBack`, which asks BEFORE allocating, since declining to
    // promote allocates nothing and so can never trigger the OOM ladder on its own
    // behalf. The second reaches here from the engine thread too, under
    // `session_mu`; see the lock-order note in the module doc.
    //
    // It does NOT take `session_mu` itself: the LLM session cannot be freed
    // underneath its own worker (every teardown path joins the worker first), and
    // taking it here would invert that order.
    fn llmForeignReclaim(ctx: *anyopaque, needed: u64) u64 {
        const self: *Driver = @ptrCast(@alignCast(ctx));
        var got: u64 = 0;
        {
            self.diff_mu.lockUncancelable(self.io);
            defer self.diff_mu.unlock(self.io);
            got = self.arbiter.requestRoom(.llm, needed);
        }
        if (got != 0) return got;
        // The lock is DROPPED before waiting, and that is the whole point: the
        // release below is enacted by the engine thread, which needs `diff_mu` to do
        // it. Waiting while holding it deadlocks until the timeout.
        return self.awaitDeferredRelease(needed);
    }

    /// The rung between "the image model accepted a release" and "the allocation
    /// fails": wait for a release the arbiter asked for to actually be enacted, and
    /// report the bytes it returned.
    ///
    /// `Diffuser.requestRelease` is a REQUEST: the free happens in `fulfillRelease`
    /// on the engine thread, because the session pointer has one writer and its
    /// readers load it unlocked. So `Arbiter.requestRoom` returns 0 for a release
    /// that was accepted and is about to happen, which is indistinguishable, to the
    /// caller, from a peer that had nothing to give. For `residency.promoteBack`
    /// that is fine: it allocated nothing and picks the room up at its next boundary
    /// poll. For the OOM ladder it is not: there is no next poll, the allocation
    /// fails now, and the turn dies.
    ///
    /// Only ever from a WORKER. On the engine thread this would wait on work only
    /// that same thread can do; its caller there (`promoteBack` under
    /// `applyMeterPolicy`) is the one that already has a next poll.
    fn awaitDeferredRelease(self: *Driver, needed: u64) u64 {
        if (std.Thread.getCurrentId() == self.engine_thread) return 0;

        // Reads under `diff_mu`, which `maybeReleaseDiffuser` holds across BOTH
        // clearing the request flag and the teardown. So every observation here is
        // of a settled state, release pending or release done, never of the
        // half-torn-down middle.
        const before = self.pendingReleaseUsage() orelse return 0;
        const deadline = std.Io.Clock.real.now(self.io).nanoseconds + release_wait_ns;
        while (std.Io.Clock.real.now(self.io).nanoseconds < deadline) {
            std.Io.sleep(self.io, .{ .nanoseconds = std.time.ns_per_ms }, .real) catch {};
            if (self.pendingReleaseUsage() != null) continue; // still queued for the engine thread
            const freed = before -| self.diffusionUsage();
            log.info("[vram] llm waited for the peer's release: {d} MiB returned (needed {d} MiB)", .{
                freed >> 20, needed >> 20,
            });
            return freed;
        }
        log.warn("[vram] llm waited {d} ms for the peer's release and it never landed; the allocation will fail", .{
            release_wait_ns / std.time.ns_per_ms,
        });
        return 0;
    }

    /// The image model's residency if a release is pending for it, else null (which
    /// covers "no engine" too: nothing is coming, so there is nothing to wait for).
    fn pendingReleaseUsage(self: *Driver) ?u64 {
        self.diff_mu.lockUncancelable(self.io);
        defer self.diff_mu.unlock(self.io);
        return self.arbiter.pendingRelease(.llm);
    }

    fn diffusionUsage(self: *Driver) u64 {
        self.diff_mu.lockUncancelable(self.io);
        defer self.diff_mu.unlock(self.io);
        const p = self.arbiter.diffusion orelse return 0;
        return p.usage();
    }

    /// Main-loop hook: enact a release the LLM asked for through the arbiter's rung
    /// (`Diffuser.requestRelease`). Runs HERE, on the engine thread, because the
    /// session pointer has exactly one writer by design and its readers load it
    /// unlocked; `diff_mu` additionally locks out the LLM worker, which reads this
    /// engine through the arbiter's participant while holding that same lock.
    fn maybeReleaseDiffuser(self: *Driver, d: *diffuser.Diffuser) void {
        if (!d.releaseRequested()) return;
        self.diff_mu.lockUncancelable(self.io);
        defer self.diff_mu.unlock(self.io);
        _ = d.fulfillRelease();
    }

    /// Main-loop hook: pump the image engine, unless the LLM is (re)loading, so no
    /// diffusion worker touches the session mid-load. The in-flight image is left
    /// running; only new starts are held.
    pub fn pumpDiffuser(self: *Driver) void {
        self.assertEngineThread();
        if (self.loading.load(.acquire)) return;
        if (self.diffuser) |*d| {
            self.maybeReleaseDiffuser(d);
            d.pump();
        }
    }

    /// `chat.Session.replan`: the LLM worker asking, at a prefill-chunk or token
    /// boundary, whether the plan now allows it more than it holds, the mirror of
    /// `llmForeignReclaim` for the case where nothing failed and so nothing reactive
    /// fires. Same thread and same lock (`diff_mu` guards the participant against a
    /// concurrent `freeDiffuser`), and, like it, deliberately NOT under
    /// `session_mu`: the session cannot be freed underneath its own worker, and
    /// taking it here would invert the documented order.
    ///
    /// Nothing under this lock can re-enter `llmForeignReclaim`, which takes the
    /// same non-reentrant mutex: `pollLlmGrowth` runs only while the LLM is busy, so
    /// its settle publishes and returns instead of promoting layers here. The
    /// promote (and any reclaim it asks for) happens after this returns, when the
    /// worker enacts the published ceiling.
    fn llmMidTurnReplan() void {
        const self = g_instance orelse return;
        self.diff_mu.lockUncancelable(self.io);
        defer self.diff_mu.unlock(self.io);
        const now = std.Io.Clock.real.now(self.io).nanoseconds;
        if (now - self.last_replan_ns < replan_interval_ns) return;
        self.last_replan_ns = now;
        _ = self.arbiter.pollLlmGrowth();
    }

    // ── The diffusion engine's configuration ──────────────────────────────────

    /// The configured checkpoint's family, from the FILE and not from the catalog.
    ///
    /// The catalog is scanned on a worker thread, so on a cold start it is empty
    /// for the first frames while the engine is already being handed its model
    /// set. Asking the catalog here meant a configured LoRA was dropped on exactly
    /// the run that had no cached index. `model_spec.Cache` memoizes by path, so
    /// this costs one header parse per checkpoint change.
    pub fn configuredFamily(self: *Driver) ?model_spec.Family {
        const path = self.settings.diffusion_model.opt() orelse return null;
        if (self.family_cache == null) self.family_cache = model_spec.Cache.init(self.gpa, self.io);
        return (self.family_cache.?.primary(path).info() orelse return null).family;
    }

    /// The LoRAs turned on for the configured checkpoint's family, as the engine
    /// takes them.
    ///
    /// Borrowed, like every path in `modelConfigFromSettings`: `requestPaths`
    /// re-dupes into the engine's own store, so nothing here outlives the frame. A
    /// disabled row is left out rather than passed at strength 0, so a session that
    /// needs no factors does not hold a gigabyte of them.
    fn loraSpecsFromSettings(self: *Driver) []const tp.pipeline.LoraSpec {
        const S = struct {
            var buf: [config.max_family_loras]tp.pipeline.LoraSpec = undefined;
        };
        const fam = self.configuredFamily() orelse return &.{};
        const key = @tagName(fam); // the same key `client/selection.zig` files LoRAs under
        var n: usize = 0;
        for (self.settings.loras.slice()) |*l| {
            if (!l.enabled or !std.mem.eql(u8, l.family.slice(), key)) continue;
            S.buf[n] = .{ .path = l.path.slice(), .strength = l.strength };
            n += 1;
        }
        return S.buf[0..n];
    }

    /// The model set the settings select, as the engine's own type. One builder,
    /// used by both the first build and the live swap, so the two cannot drift as
    /// components are added. "" = not overridden; the pipeline resolves that
    /// component out of the primary checkpoint. Never substitute a default here: a
    /// defaulted path that reaches `Options` is indistinguishable from a user
    /// request.
    fn modelConfigFromSettings(self: *Driver) diffuser.ModelConfig {
        return .{
            .dit_path = self.settings.diffusion_model.opt().?,
            .vae_path = self.settings.vae.slice(),
            .text_encoder_path = self.settings.text_encoder.slice(),
            .text_encoder_2_path = self.settings.text_encoder_2.slice(),
            .backend = pipeline_map.toPipelineBackend(self.settings.diff_backend),
            .vae_decode = pipeline_map.toPipelineVae(self.settings.vae_decode),
            .loras = self.loraSpecsFromSettings(),
        };
    }

    fn diffConfigFromSettings(self: *Driver) diffuser.DiffConfig {
        const m = self.modelConfigFromSettings();
        const c = &self.settings;
        return .{
            .dit_path = m.dit_path,
            .vae_path = m.vae_path,
            .text_encoder_path = m.text_encoder_path,
            .text_encoder_2_path = m.text_encoder_2_path,
            .steps = c.steps,
            .width = c.width,
            .height = c.height,
            .sampler = pipeline_map.toPipelineSampler(c.sampler),
            .scheduler = pipeline_map.toPipelineScheduler(c.scheduler),
            .prompt_syntax = pipeline_map.toPipelineSyntax(c.prompt_syntax),
            .emphasis = pipeline_map.toPipelineEmphasis(c.emphasis),
            .compat = pipeline_map.toPipelineCompat(c.compat),
            .backend = m.backend,
            .vae_decode = m.vae_decode,
            .preview_enabled = c.preview != .none,
            .taew_path = if (c.preview == .taesd) c.taesd.opt() else null,
            .preview_ds = c.taesd_size.divisor(),
        };
    }

    /// Reconcile the image engine with `settings`: build it when a diffusion model
    /// is (newly) configured, free it when cleared, and push path/default/preview
    /// changes into a live one (a model swap defers until the queue is idle).
    pub fn syncDiffuser(self: *Driver) void {
        self.assertEngineThread();
        const c = &self.settings;
        if (c.diffusion_model.opt() == null) {
            self.freeDiffuser();
            return;
        }
        if (self.diffuser == null) {
            self.diffuser = diffuser.Diffuser.init(self.gpa, self.io, self.wake, self.diffConfigFromSettings(), self.coordinator());
            // Before anything can be enqueued: an image asks the coordinator for
            // its budget the moment it starts, and the main-loop hook is one
            // iteration too late for the FIRST one, which is the image most
            // likely to be the one that does not fit.
            defer self.applyDiffusionOnlyPolicy();
            self.diffuser.?.seedBase(@truncate(@as(u96, @bitCast(std.Io.Clock.real.now(self.io).nanoseconds))));
            // Carry the previously MEASURED peak residency across sessions, so the
            // first image of a run sizes the LLM's eviction by what this pipeline
            // actually costs instead of by what its checkpoints weigh. Keyed on the
            // DiT path: a different model invalidates the figure.
            if (c.diff_peak_resident != 0) {
                if (c.diffusion_model.opt()) |m| {
                    if (config.modelKey(m) == c.diff_peak_key)
                        self.diffuser.?.seedPeakResident(c.diff_peak_resident);
                }
            }
            // Register the image model as an arbiter participant so a growing LLM can
            // reclaim its idle residency. The participant borrows the engine;
            // `freeDiffuser` drops it before teardown. `diff_mu` because the LLM's
            // worker thread reads it (foreignReclaim).
            self.diff_mu.lockUncancelable(self.io);
            self.arbiter.diffusion = self.diffuser.?.participant();
            self.diff_mu.unlock(self.io);
        }
        var d = &self.diffuser.?;
        // requestPaths re-dupes the paths into the engine's owned store (nothing
        // aliases the settings buffers) and applies/swaps once the queue is idle.
        d.requestPaths(
            self.modelConfigFromSettings(),
            if (c.preview == .taesd) c.taesd.opt() else null,
        );
        d.setDefaults(c.steps, c.width, c.height);
        d.setSampler(pipeline_map.toPipelineSampler(c.sampler));
        d.setScheduler(pipeline_map.toPipelineScheduler(c.scheduler));
        d.setPromptSyntax(
            pipeline_map.toPipelineSyntax(c.prompt_syntax),
            pipeline_map.toPipelineEmphasis(c.emphasis),
            pipeline_map.toPipelineCompat(c.compat),
        );
        d.setPreview(c.preview);
        d.setPreviewSize(c.taesd_size.divisor());
    }

    /// Tear down the image engine (diffusion model cleared, or at exit): cancel any
    /// in-flight/queued generation so the worker aborts instead of blocking the
    /// join, then free it (frees the whole image history it owns). Nothing else has
    /// to be told: every holder keeps ids, which stop resolving.
    pub fn freeDiffuser(self: *Driver) void {
        if (self.diffuser) |*d| {
            // Drop the arbiter's participant FIRST: it borrows the engine we are about
            // to free, and the LLM's worker thread can be reading it right now through
            // `llmForeignReclaim`. Mirrors `arbiter.llm = null` on the LLM side.
            // `diff_active` goes with it, we tear down without pumping, so the
            // queue-drain edge that normally clears it never fires, and a stale `true`
            // would keep the LLM pinned to its share with nothing left to use the rest.
            self.diff_mu.lockUncancelable(self.io);
            self.arbiter.diffusion = null;
            self.arbiter.diff_active = false;
            self.diff_mu.unlock(self.io);
            d.cancelAll();
            d.deinit();
            self.diffuser = null;
        }
    }

    // ── Settings ──────────────────────────────────────────────────────────────

    /// Bring the engines in line with `new`: rebuild or retune the diffuser, reload
    /// the LLM when its model set changed, else push the live settings. Then `new`
    /// becomes `settings`. A change that alters the LLM load or the image-tool
    /// availability (which changes the system prompt) forces a transcript-preserving
    /// reload, but only if the LLM is resident; if it has not lazy-loaded yet, the
    /// new settings are simply picked up on the first message.
    pub fn applySettings(self: *Driver, new: *const config.Config) void {
        self.assertEngineThread();
        const old = &self.settings;
        // A system-prompt edit applies live on a render-driven (template) session:
        // updateSettings restages it and the next message re-renders. A hand-glue
        // session bakes the system into the prompt prefix, so it needs a
        // transcript-preserving reload instead.
        const sys_changed = !std.mem.eql(u8, new.system_prompt.slice(), old.system_prompt.slice());
        const sys_needs_reload = sys_changed and self.session != null and
            !self.loading.load(.acquire) and !self.session.?.templateActive();
        // The same shape for the image tool: a template session picks the flag
        // up live in `updateSettings`, a hand-glue one has it baked into the
        // prompt prefix and needs the transcript-preserving reload.
        const tool_needs_reload = new.image_tool != old.image_tool and self.session != null and
            !self.loading.load(.acquire) and !self.session.?.templateActive();
        const llm_reload = !new.llmReloadEql(old) or
            (new.diffEnabled() != old.diffEnabled()) or // tool prompt changes
            sys_needs_reload or tool_needs_reload;
        // A KV-dtype change needs only a CONTEXT rebuild (weights stay resident),
        // never the full weight reload above.
        const ctx_reload = !new.ctxReloadEql(old);
        self.settings = new.*;
        self.adoptMeterFractions(new);

        // The diffusion engine is shared by both modes; reconcile it either way.
        self.syncDiffuser();

        if (self.session != null) {
            if (llm_reload) {
                self.reload_requested = true; // transcript-preserving (see loaderMain)
            } else if (!self.loading.load(.acquire)) {
                const s = self.session.?;
                s.updateSettings(&self.settings); // reasoning / VRAM priority, live
                // Weight noise lands in a table the decode kernels re-read every
                // launch, so a change takes effect on the next token, mid-reply.
                s.be.weight_noise.amount = self.settings.weight_noise_amount;
                s.be.weight_noise.setCurve(if (self.settings.weight_noise) self.settings.weight_noise_curve.slice() else "");
                if (ctx_reload) s.rebuildContext(chat.toKvDtype(self.settings.kv_dtype)) catch |err|
                    log.err("kv-dtype context rebuild failed: {t}", .{err});
            }
        }
        // Governs LOADING, so no reload is forced: the next model load picks it up.
        applyWeightRead(self.settings.weight_read);
    }

    /// A live-only change (the reasoning toggles): adopt `new` and push it into the
    /// running session without any reconciliation.
    pub fn updateSettingsLive(self: *Driver, new: *const config.Config) void {
        self.settings = new.*;
        if (self.uiSession()) |s| s.updateSettings(&self.settings);
    }

    /// Push the checkpoint-read setting to the core global every loader consults.
    /// It governs LOADING, so the effect lands on the next model load rather than
    /// on anything resident.
    pub fn applyWeightRead(w: config.WeightRead) void {
        tp.safetensors.read_mode = switch (w) {
            .pread => .pread,
            .mmap => .mmap,
            .buffered => .buffered,
        };
    }

    // ── Turns ─────────────────────────────────────────────────────────────────

    /// Send a message through the session, or stash it and lazy-load the model.
    /// Returns false when it went nowhere (empty after trimming, no model
    /// configured, the session refused it, allocation failed), so a headless
    /// driver is not left waiting forever.
    pub fn submit(self: *Driver, text: []const u8) bool {
        self.assertEngineThread();
        const trimmed = std.mem.trim(u8, text, " \t\r\n");
        if (trimmed.len == 0) return false;
        // Never touch the session while a (re)load is in flight, the loader thread is
        // freeing/rebuilding it. Submit only when it's live; otherwise stash the text
        // and the load-completion path (maybeStartReload) auto-submits it.
        if (self.uiSession()) |s| {
            s.submit(trimmed) catch |err| {
                log.err("submit failed: {t}", .{err});
                return false;
            };
            // submit itself no-ops on a turn already in flight or queued.
            return s.busy() or s.turnPending();
        }
        if (self.settings.llm_model.opt() == null) return false; // no model to load
        if (self.pending_submit) |p| self.gpa.free(p);
        self.pending_submit = self.gpa.dupe(u8, trimmed) catch null;
        if (self.pending_submit == null) return false; // nothing stashed, so don't reload for it
        if (!self.loading.load(.acquire)) self.reload_requested = true; // else the in-flight load will pick it up
        return true;
    }

    /// Regenerate asked for while the LLM is unloaded: lazy-load it and regenerate
    /// the carried transcript's last reply once the fresh session adopts it (see
    /// maybeStartReload). No-op if no model is configured.
    pub fn requestRegenLoad(self: *Driver) void {
        self.assertEngineThread();
        if (self.settings.llm_model.opt() == null) return; // no model to load
        self.pending_regenerate = true;
        if (!self.loading.load(.acquire)) self.reload_requested = true; // else the in-flight load picks it up
    }

    /// Start a fresh conversation. Only the transcript is reset; generated images
    /// live in the engine's shared history and stay in the gallery. While a turn
    /// is generating or a load runs it waits (`maybeApplyPendingTranscript`), so a
    /// `chat_new` right behind a `chat_cancel` lands once the worker stops.
    pub fn newChat(self: *Driver) void {
        self.assertEngineThread();
        if (!self.transcriptIdle()) return self.setPendingTranscript(.new);
        self.newChatNow();
    }

    fn newChatNow(self: *Driver) void {
        if (self.session) |s| _ = s.reset();
        // With the LLM ejected the transcript lives in `carry`.
        self.freeCarry();
        self.freeLlmSuspend(); // a new chat never resumes a suspended turn
        self.pending_regenerate = false;
        if (self.llm_paused) { // a fresh chat starts unpaused
            self.llm_paused = false;
            if (self.uiSession()) |s| s.setPaused(false);
        }
        self.clearStaged(); // images staged for a not-yet-loaded first message
        self.dropNotes(); // the conversation they were about is being left
    }

    /// Replace the transcript with `msgs` (a stored conversation): into the live
    /// session, replaying every turn into its tokenizer, or into the carry when
    /// nothing is resident, where the next session adopts it. Takes ownership of
    /// `msgs`. While a turn is in flight or a load runs it waits, like `newChat`:
    /// the adopt would swap `messages` out from under a worker rendering its
    /// prompt from them.
    pub fn adoptTranscript(self: *Driver, msgs: std.ArrayList(chat.Message)) void {
        self.assertEngineThread();
        if (!self.transcriptIdle()) return self.setPendingTranscript(.{ .adopt = msgs });
        self.adoptTranscriptNow(msgs);
    }

    fn adoptTranscriptNow(self: *Driver, msgs: std.ArrayList(chat.Message)) void {
        self.freeCarry();
        self.freeLlmSuspend();
        self.clearStaged();
        self.pending_regenerate = false;
        if (self.session) |s| {
            _ = s.reset(); // takes: `transcriptIdle` already ruled out busy
            s.adoptTranscript(msgs) catch |err| log.err("history: adopting conversation failed: {t}", .{err});
        } else {
            self.carry_mu.lockUncancelable(self.io);
            defer self.carry_mu.unlock(self.io);
            self.carry = msgs;
        }
    }

    /// The transcript is this thread's to replace: no load in flight and no
    /// worker rendering a prompt from it.
    fn transcriptIdle(self: *Driver) bool {
        if (self.loading.load(.acquire)) return false;
        if (self.session) |s| if (s.busy()) return false;
        return true;
    }

    /// The latest request wins; an adopt it replaces is freed.
    fn setPendingTranscript(self: *Driver, p: PendingTranscript) void {
        self.dropPendingTranscript();
        self.pending_transcript = p;
    }

    fn dropPendingTranscript(self: *Driver) void {
        const p = self.pending_transcript orelse return;
        self.pending_transcript = null;
        switch (p) {
            .new => {},
            .adopt => |m| {
                var list = m;
                for (list.items) |*msg| msg.deinit(self.gpa);
                list.deinit(self.gpa);
            },
        }
    }

    /// Main-loop hook: land a deferred `newChat` / `adoptTranscript` once the
    /// worker has stopped and any load has finished. After the session's `poll`,
    /// which is what joins a finished worker.
    pub fn maybeApplyPendingTranscript(self: *Driver) void {
        self.assertEngineThread();
        const p = self.pending_transcript orelse return;
        if (!self.transcriptIdle()) return;
        self.pending_transcript = null;
        switch (p) {
            .new => self.newChatNow(),
            .adopt => |m| self.adoptTranscriptNow(m),
        }
    }

    // ── Load and reload ───────────────────────────────────────────────────────

    /// Main-loop hook: reap a finished loader, and start a pending (re)load when
    /// none is in flight. Runs on the engine thread so the loading-flag hand-off to
    /// the loader is well-ordered. `before_load` runs just before a load is armed,
    /// while this thread still owns the transcript (the owner flushes its history
    /// there).
    pub fn maybeStartReload(self: *Driver, before_load: *const fn () void) void {
        self.assertEngineThread();
        if (self.loader) |t| {
            if (!self.loading.load(.acquire)) {
                t.join();
                self.loader = null;
                // Load finished. Apply the current meter policy to the fresh session
                // (settles the LLM to its share if diffusion is already resident).
                if (self.session != null) self.applyMeterPolicy();
                // Move any images staged before the lazy load into the fresh session
                // (BEFORE the deferred submit, so the first message carries them). If
                // the load failed (no session) or the model has no vision tower, drop
                // them, attachImage no-ops without a tower.
                if (self.session) |s| {
                    for (self.staged.items) |st|
                        s.attachImage(st.rgb, st.width, st.height) catch |err| log.err("attach staged image: {t}", .{err});
                }
                self.clearStaged();
                // If the first message was stashed while the LLM lazy-loaded, submit
                // it now that the session is live.
                if (self.pending_submit) |text| {
                    self.pending_submit = null;
                    if (self.session) |s| s.submit(text) catch |err| log.err("deferred submit: {t}", .{err});
                    self.gpa.free(text);
                }
                // Resume-continue a mid-turn response suspended by unload-while-paused:
                // `ids` was restored verbatim; continue decoding it.
                if (self.pending_continue) {
                    self.pending_continue = false;
                    if (self.session) |s| s.continueOpenTurn() catch |err| log.err("resume continue: {t}", .{err});
                }
                // Regenerate requested while unloaded: the fresh session has adopted
                // the carried transcript, so regenerate its last reply now.
                if (self.pending_regenerate) {
                    self.pending_regenerate = false;
                    if (self.session) |s| s.regenerate() catch |err| log.err("deferred regenerate: {t}", .{err});
                }
            }
        }
        if (!self.reload_requested or self.loading.load(.acquire) or self.loader != null) return;
        // Paused with nothing resident: HOLD the load. A message / regenerate / turn
        // queued while paused keeps `reload_requested` set but loads NOTHING until
        // the user resumes (toggleLlmPause kicks this the moment the gate lifts),
        // mirroring the diffusion side, where Diffuser.pump defers loads while paused.
        if (self.llm_paused and self.session == null) return;
        // The LLM (re)loads CONCURRENTLY with diffusion, a running image keeps
        // generating on its own context while the LLM builds on a fresh one, so a
        // chat sent mid-image loads and responds right away instead of waiting for
        // the image. The only shared state is the session pointer, which the loader's
        // teardown/publish and the worker-thread coordinator hooks serialize with
        // `session_mu`. The pump stays gated while `loading` is set, so no NEW image
        // starts mid-load; the in-flight one is left running (not reaped).
        self.reload_requested = false;
        self.load_err = null;
        before_load();
        self.load_settings = self.settings;
        self.loading.store(true, .release);
        self.loader = std.Thread.spawn(.{}, loaderMain, .{self}) catch |err| {
            log.err("spawn loader failed: {t}", .{err});
            self.load_err = err;
            self.loading.store(false, .release);
            return;
        };
    }

    /// Background (re)load: tear down the old session, then build a new one from
    /// `load_settings`. Runs on its own thread, creates the CUDA context there; the
    /// generation/diffusion workers bind to it as before. On completion it publishes
    /// `session` and clears `loading` (release) so the engine thread can adopt it.
    fn loaderMain(self: *Driver) void {
        // Tear down the previous session first. Its CUDA context must be current on
        // this thread to free device memory, so bind it before deinit. Before
        // freeing it, stop the LLM turn but let any in-flight image FINISH (don't
        // cancel it), then detach the transcript so the chat survives the swap.
        if (self.session) |s| {
            s.be.bindThread();
            s.requestCancel();
            if (s.worker) |t| {
                t.join();
                s.worker = null;
            }
            // Diffusion may be generating concurrently (its worker reads the session
            // via the coordinator hooks), so serialize the teardown with
            // `session_mu`, the same guard unloadLlm uses.
            self.session_mu.lockUncancelable(self.io);
            {
                // The owner renders this the moment it lands, so publish it as one
                // step rather than letting a frame catch a half-written optional.
                self.carry_mu.lockUncancelable(self.io);
                defer self.carry_mu.unlock(self.io);
                self.carry = s.detachTranscript();
            }
            s.deinit();
            self.session = null;
            self.arbiter.llm = null; // participant points into the freed session
            self.session_mu.unlock(self.io);
        }
        if (self.session_arena) |a| {
            a.deinit();
            self.gpa.destroy(a);
            self.session_arena = null;
        }

        if (self.load_settings.llm_model.opt() == null) {
            // Nothing to load (LLM cleared): drop the carried transcript and leave
            // the notice showing.
            self.freeCarry();
            self.freeLlmSuspend();
            self.loading.store(false, .release);
            self.wake();
            return;
        }

        const arena_obj = self.gpa.create(std.heap.ArenaAllocator) catch |err| return self.finishLoad(err);
        arena_obj.* = std.heap.ArenaAllocator.init(self.gpa);
        self.session_arena = arena_obj;

        const t0 = std.Io.Clock.real.now(self.io).nanoseconds;
        const s = self.buildSession(arena_obj.allocator()) catch |err| {
            log.err("failed to load session: {t}", .{err});
            arena_obj.deinit();
            self.gpa.destroy(arena_obj);
            self.session_arena = null;
            self.freeCarry(); // no session to adopt the transcript
            self.freeLlmSuspend();
            return self.finishLoad(err);
        };
        // Replay the carried transcript into the new model (KV empty; the next turn's
        // prefill replays it) so a model swap keeps the chat.
        // The hold spans the ADOPT, not just the clear: `adoptTranscript` moves the
        // list into the new session on its first line, so a frame that read `carry`
        // between that move and the clear would walk messages the session now owns
        // (and is re-tokenizing). A few ms of a blocked render is the price.
        self.carry_mu.lockUncancelable(self.io);
        if (self.carry) |m| {
            s.be.bindThread();
            if (self.llm_suspend) |sus| {
                // Unload-while-paused resume: restore the exact `ids` (open turn)
                // verbatim instead of replaying (which would close the turn), then
                // continue decoding that response after publish.
                s.adoptSuspended(m, sus.ids) catch |err| log.err("adopt suspended failed: {t}", .{err});
                if (sus.midturn) self.pending_continue = true;
                self.gpa.free(sus.ids);
                self.llm_suspend = null;
            } else {
                s.adoptTranscript(m) catch |err| log.err("adopt transcript failed: {t}", .{err});
            }
            self.carry = null;
        }
        self.carry_mu.unlock(self.io);
        // A paused reload that is NOT a resume (e.g. a backend switch while paused)
        // starts the fresh gate paused so the state matches the button.
        if (self.llm_paused) s.pause.pause(self.io);
        const dt = @as(f64, @floatFromInt(std.Io.Clock.real.now(self.io).nanoseconds - t0)) / 1e9;
        log.info("[vram] LLM session loaded/ready in {d:.1}s", .{dt});
        // Publish under the lock: the diffusion worker's coordinator hooks read
        // `session` and must see either null or a fully-built session, never a tear.
        self.session_mu.lockUncancelable(self.io);
        self.session = s;
        self.arbiter.llm = s.participant();
        s.replan = llmMidTurnReplan; // re-plan at the worker's own boundaries
        self.session_mu.unlock(self.io);
        self.finishLoad(null);
    }

    /// Free a carried transcript that has no destination (LLM cleared or load
    /// failed). Messages are gpa-owned.
    /// Text for the model to read at the next turn boundary. A session that is
    /// not resident yet keeps it here rather than losing it.
    pub fn queueNote(self: *Driver, text: []u8) void {
        self.assertEngineThread();
        if (self.uiSession()) |s| return s.queueNote(text);
        self.pending_notes.append(self.gpa, text) catch self.gpa.free(text);
    }

    /// Hand over what arrived while nothing was resident.
    pub fn flushNotesTo(self: *Driver, s: *chat.Session) void {
        for (self.pending_notes.items) |n| s.queueNote(n);
        self.pending_notes.clearRetainingCapacity();
    }

    fn dropNotes(self: *Driver) void {
        for (self.pending_notes.items) |n| self.gpa.free(n);
        self.pending_notes.clearRetainingCapacity();
    }

    pub fn freeCarry(self: *Driver) void {
        self.carry_mu.lockUncancelable(self.io);
        defer self.carry_mu.unlock(self.io);
        if (self.carry) |*m| {
            for (m.items) |*msg| msg.deinit(self.gpa);
            m.deinit(self.gpa);
            self.carry = null;
        }
    }

    /// Publish the load result: record any error, then clear the loading flag
    /// (release) as the last write so the engine thread's acquire read sees a
    /// settled `session`.
    fn finishLoad(self: *Driver, err: ?anyerror) void {
        self.load_err = err;
        self.loading.store(false, .release);
        self.wake();
    }

    /// Build the LLM session from `load_settings`. Vision needs the tower; the
    /// image tool is available when a diffusion model is configured. Paths are
    /// duped into `arena` so the session never aliases the settings buffers.
    fn buildSession(self: *Driver, arena: std.mem.Allocator) !*chat.Session {
        const seed: u64 = @truncate(@as(u96, @bitCast(std.Io.Clock.real.now(self.io).nanoseconds)));
        const s = try chat.Session.init(arena, self.gpa, self.io, self.wake, try chat.sessionOptions(arena, &self.load_settings, seed));
        s.be.foreign_reclaim = .{ .ctx = @ptrCast(self), .call = llmForeignReclaim };
        return s;
    }
};
