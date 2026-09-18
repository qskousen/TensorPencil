//! Which host should take a render: a pure function of facts each host already
//! tells this client (catalog, state block, telemetry, and what its own
//! finished images cost). `client/hosts.zig` holds the queue and decides when
//! to hand a job over; nothing here touches a socket.
//!
//! The estimate is seconds until the job is done there: a pipeline load when
//! none is resident, streaming when the card cannot hold the model, and what
//! is already running or waiting there. A busy host can win, in which case the
//! caller holds the job for it. A card too small for the model is a cost and
//! never a veto: the engine streams what it cannot keep resident.
//!
//! The measured tables live in memory only. This client writes no record of
//! what was rendered or how long it took.
const std = @import("std");

/// ASSUMED, not measured: a step on a host that has never finished an image,
/// per megapixel. One finished image replaces it with that host's own figure.
pub const assumed_s_per_step_per_mpx: f64 = 1.5;
/// ASSUMED: bringing a pipeline in when none is resident.
pub const assumed_load_s: f64 = 20;
/// ASSUMED: rendering beside a chat costs something. Both of these are guesses
/// with no measurement behind them; the table tests must not depend on either.
pub const chat_contention: f64 = 1.15;
/// ASSUMED: step cost multiplier when none of the weights fit and every step
/// streams them over PCIe, scaled by the fraction that misses.
pub const stream_penalty: f64 = 3.0;
/// Below this, the seconds before the first step were not a load: the pipeline
/// was already resident and the figure would poison the cold-load estimate.
pub const cold_load_floor_s: f64 = 3.0;

pub const HostId = u32;

/// One render, as the client knows it before anybody has accepted it.
pub const Job = struct {
    /// Catalog family tag of the checkpoint, "" when unknown.
    family: []const u8 = "",
    /// Megapixels of the requested image.
    mpx: f64 = 1,
    steps: u32 = 20,
    /// What it would hold on a card with room to spare, 0 when unknown.
    vram_need: u64 = 0,
};

/// What one host looks like to the scheduler. Every field is something the
/// client already has; nothing here is new on the wire.
pub const Host = struct {
    id: HostId,
    /// Connected and answering.
    up: bool = false,
    /// It has an image engine and is not paused.
    ready: bool = false,
    /// Its catalog holds the job's model.
    has_model: bool = false,
    /// It carries the chat.
    chat: bool = false,
    /// Bytes the image engine may use there: the cap or the card, less what
    /// the chat model holds. 0 = it did not say.
    vram_limit: u64 = 0,
    /// Renders waiting there behind the running one, including what this
    /// client handed it and it has not listed yet.
    queued: u32 = 0,
    /// A render is running there now.
    busy: bool = false,
    /// Seconds left on that render, 0 when unknown.
    busy_remaining_s: f64 = 0,
    /// Family of the pipeline resident now, "" when none is.
    resident_family: []const u8 = "",
    /// Measured on this host, 0 when it has finished nothing yet.
    s_per_step: f64 = 0,
    load_s: f64 = 0,
};

pub const Verdict = union(enum) {
    /// Send it here.
    place: HostId,
    /// Nobody holds the model; this host could run it once it does.
    send_model: HostId,
    /// Nowhere to put it: nothing is up, or it fits nowhere.
    nowhere,
};

/// Seconds until `j` is done on this host, or null when it cannot take the job
/// at all: down, no image engine, or it does not hold the model.
pub fn estimate(h: *const Host, j: Job) ?f64 {
    if (!h.up or !h.ready or !h.has_model) return null;
    return cost(h, j);
}

/// How much of the job's footprint this card cannot keep resident, 0..1. The
/// overflow streams over PCIe per step rather than failing.
fn streamedFraction(h: *const Host, j: Job) f64 {
    if (h.vram_limit == 0 or j.vram_need == 0) return 0;
    if (j.vram_need <= h.vram_limit) return 0;
    const over: f64 = @floatFromInt(j.vram_need - h.vram_limit);
    return @min(over / @as(f64, @floatFromInt(j.vram_need)), 1.0);
}

fn cost(h: *const Host, j: Job) f64 {
    const base = if (h.s_per_step > 0) h.s_per_step else assumed_s_per_step_per_mpx * j.mpx;
    // A measured host is charged the streaming penalty too: the measurement
    // describes the job it ran, not this one.
    const per_step = base * (1.0 + streamedFraction(h, j) * (stream_penalty - 1.0));
    const resident = j.family.len > 0 and std.mem.eql(u8, h.resident_family, j.family);
    const load: f64 = if (resident) 0 else (if (h.load_s > 0) h.load_s else assumed_load_s);
    const job = per_step * @as(f64, @floatFromInt(j.steps));
    const contend: f64 = if (h.chat) chat_contention else 1.0;
    // What is there already runs first; jobs waiting are taken to be like this one.
    const running: f64 = if (!h.busy) 0 else if (h.busy_remaining_s > 0) h.busy_remaining_s else job;
    const ahead = running + job * @as(f64, @floatFromInt(h.queued));
    return (load + job) * contend + ahead;
}

/// The host that finishes `j` soonest, busy or not. When none holds the
/// model, the one to send it to.
pub fn place(hosts: []const Host, j: Job) Verdict {
    var best: ?HostId = null;
    var best_cost: f64 = std.math.inf(f64);
    for (hosts) |*h| {
        const c = estimate(h, j) orelse continue;
        if (c < best_cost) {
            best_cost = c;
            best = h.id;
        }
    }
    if (best) |id| return .{ .place = id };

    // Nobody has the file. Any host that is up could run it once it does;
    // `ready` is not required, since a host with no checkpoint it can resolve
    // has no image engine yet for exactly that reason.
    var alt: ?HostId = null;
    var alt_cost: f64 = std.math.inf(f64);
    for (hosts) |*h| {
        if (!h.up or h.has_model) continue;
        const c = cost(h, j);
        if (c < alt_cost) {
            alt_cost = c;
            alt = h.id;
        }
    }
    if (alt) |id| return .{ .send_model = id };
    return .nowhere;
}

// ── What a host was measured at ───────────────────────────────────────────

pub const max_families: usize = 12;
pub const max_family_tag: usize = 24;

/// One host's own figures, per family: what a step cost per megapixel, and
/// what a cold load cost. In memory only, and reset with the host.
pub const Table = struct {
    rows: [max_families]Row = @splat(.{}),
    n: usize = 0,

    pub const Row = struct {
        tag: [max_family_tag]u8 = .{0} ** max_family_tag,
        tag_len: u8 = 0,
        /// Seconds per step per megapixel: a step is taken to scale with area,
        /// which is first-order right for every family here.
        s_per_mpx: f64 = 0,
        steps_seen: u32 = 0,
        load_s: f64 = 0,
        loads_seen: u32 = 0,

        pub fn family(r: *const Row) []const u8 {
            return r.tag[0..r.tag_len];
        }
    };

    /// One finished image, straight from its `wire.ImageInfo` fields. `steps`
    /// is what ran; the timestamps are the engine's. Returns false when the
    /// image says nothing useful (cancelled, one step, no timings).
    pub const Sample = struct {
        family: []const u8 = "",
        mpx: f64 = 0,
        steps: u32 = 0,
        start_ns: i64 = 0,
        first_step_ns: i64 = 0,
        last_step_ns: i64 = 0,
    };

    pub fn note(self: *Table, s: Sample) bool {
        if (s.family.len == 0 or s.mpx <= 0 or s.steps < 2) return false;
        if (s.first_step_ns <= 0 or s.last_step_ns <= s.first_step_ns) return false;
        const ns_per_s: f64 = @floatFromInt(std.time.ns_per_s);
        // The first callback lands after step 1, so the span covers steps-1.
        const span_s = @as(f64, @floatFromInt(s.last_step_ns - s.first_step_ns)) / ns_per_s;
        const per_step = span_s / @as(f64, @floatFromInt(s.steps - 1));
        const row = self.rowFor(s.family) orelse return false;
        row.steps_seen += 1;
        row.s_per_mpx = ewma(row.s_per_mpx, per_step / s.mpx, row.steps_seen);

        // Time before the first step is a load only when it was long enough to
        // be one; otherwise the pipeline was already there.
        if (s.start_ns > 0 and s.first_step_ns > s.start_ns) {
            const pre_s = @as(f64, @floatFromInt(s.first_step_ns - s.start_ns)) / ns_per_s;
            if (pre_s >= cold_load_floor_s) {
                row.loads_seen += 1;
                row.load_s = ewma(row.load_s, pre_s, row.loads_seen);
            }
        }
        return true;
    }

    /// Seconds a step of `mpx` should cost here, 0 when this host has not run
    /// that family yet.
    pub fn stepCost(self: *const Table, family: []const u8, mpx: f64) f64 {
        const row = self.find(family) orelse return 0;
        if (row.steps_seen == 0) return 0;
        return row.s_per_mpx * mpx;
    }

    /// Seconds a cold load of `family` cost here, 0 when none was seen.
    pub fn loadCost(self: *const Table, family: []const u8) f64 {
        const row = self.find(family) orelse return 0;
        return if (row.loads_seen == 0) 0 else row.load_s;
    }

    fn find(self: *const Table, family: []const u8) ?*const Row {
        for (self.rows[0..self.n]) |*r| if (std.mem.eql(u8, r.family(), family)) return r;
        return null;
    }

    /// The row for `family`, adding one while there is room.
    fn rowFor(self: *Table, family: []const u8) ?*Row {
        for (self.rows[0..self.n]) |*r| if (std.mem.eql(u8, r.family(), family)) return r;
        if (self.n == self.rows.len or family.len > max_family_tag) return null;
        const r = &self.rows[self.n];
        self.n += 1;
        @memcpy(r.tag[0..family.len], family);
        r.tag_len = @intCast(family.len);
        return r;
    }

    /// Averages the first few samples outright, then follows the recent ones:
    /// a host that changed backend should not be described by last week.
    fn ewma(old: f64, sample: f64, n: u32) f64 {
        if (n <= 1) return sample;
        const a: f64 = @max(0.25, 1.0 / @as(f64, @floatFromInt(n)));
        return old * (1 - a) + sample * a;
    }
};

// ── Tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;

test "a job goes to the host that will finish it first, and a full host is refused outright" {
    const gb: u64 = 1 << 30;
    const j: Job = .{ .family = "krea2", .mpx = 1, .steps = 20, .vram_need = 8 * gb };

    // Same speed, but one already holds the pipeline: the load decides.
    var hosts = [_]Host{
        .{ .id = 0, .up = true, .ready = true, .has_model = true, .vram_limit = 24 * gb, .s_per_step = 1 },
        .{ .id = 1, .up = true, .ready = true, .has_model = true, .vram_limit = 24 * gb, .s_per_step = 1, .resident_family = "krea2" },
    };
    try testing.expectEqual(Verdict{ .place = 1 }, place(&hosts, j));

    // A queue in front outweighs residency once it is long enough.
    hosts[1].queued = 3;
    try testing.expectEqual(Verdict{ .place = 0 }, place(&hosts, j));
    hosts[1].queued = 0;

    // A busy host about to finish beats an idle one that is slower: the
    // caller holds the job for it.
    hosts[1].busy = true;
    hosts[1].busy_remaining_s = 5;
    hosts[0].s_per_step = 3;
    try testing.expectEqual(Verdict{ .place = 1 }, place(&hosts, j));
    // With no idea how long it has left, a whole job is assumed.
    hosts[1].busy_remaining_s = 0;
    try testing.expectEqual(Verdict{ .place = 1 }, place(&hosts, j));
    hosts[0].s_per_step = 0.5;
    try testing.expectEqual(Verdict{ .place = 0 }, place(&hosts, j));
    hosts[0].s_per_step = 1;
    hosts[1].busy = false;

    // A card too small for the model is a slower host, never an excluded one.
    // Given measured figures, the measurement decides, small card or not.
    var small = [_]Host{
        .{ .id = 0, .up = true, .ready = true, .has_model = true, .vram_limit = 4 * gb, .s_per_step = 0.1 },
        .{ .id = 1, .up = true, .ready = true, .has_model = true, .vram_limit = 24 * gb, .s_per_step = 5, .queued = 2 },
    };
    try testing.expectEqual(Verdict{ .place = 0 }, place(&small, j));
    // With nothing measured anywhere, the card that holds the job wins on the
    // streaming penalty alone.
    var cold = [_]Host{
        .{ .id = 0, .up = true, .ready = true, .has_model = true, .vram_limit = 4 * gb },
        .{ .id = 1, .up = true, .ready = true, .has_model = true, .vram_limit = 24 * gb },
    };
    try testing.expectEqual(Verdict{ .place = 1 }, place(&cold, j));
    // ...and when the small card is the ONLY one, it still gets the job.
    var only = [_]Host{.{ .id = 3, .up = true, .ready = true, .has_model = true, .vram_limit = 2 * gb }};
    try testing.expectEqual(Verdict{ .place = 3 }, place(&only, j));

    // A host that is down, or has no image engine, is not a candidate.
    var off = [_]Host{
        .{ .id = 0, .up = false, .ready = true, .has_model = true },
        .{ .id = 1, .up = true, .ready = false, .has_model = true },
    };
    try testing.expectEqual(Verdict.nowhere, place(&off, j));
}

test "a cold host is ranked by what it would cost, not treated as free" {
    const j: Job = .{ .family = "sd15", .mpx = 1, .steps = 10 };
    // One host measured and loaded; one that has never run anything. The cold
    // one must not win on its (absent) numbers alone.
    var hosts = [_]Host{
        .{ .id = 0, .up = true, .ready = true, .has_model = true, .s_per_step = 1, .resident_family = "sd15" },
        .{ .id = 1, .up = true, .ready = true, .has_model = true },
    };
    try testing.expectEqual(Verdict{ .place = 0 }, place(&hosts, j));
    // Measured as much faster, the cold host wins once its load is paid off.
    hosts[1].s_per_step = 0.05;
    hosts[1].load_s = 1;
    try testing.expectEqual(Verdict{ .place = 1 }, place(&hosts, j));
    // The chat host is the tie-break when everything else matches.
    var tied = [_]Host{
        .{ .id = 0, .up = true, .ready = true, .has_model = true, .s_per_step = 1, .resident_family = "sd15", .chat = true },
        .{ .id = 1, .up = true, .ready = true, .has_model = true, .s_per_step = 1, .resident_family = "sd15" },
    };
    try testing.expectEqual(Verdict{ .place = 1 }, place(&tied, j));
}

test "nobody holding the model names the host to send it to, and a host that cannot fit it is not named" {
    const gb: u64 = 1 << 30;
    const j: Job = .{ .family = "krea2", .mpx = 1, .steps = 20, .vram_need = 12 * gb };
    var hosts = [_]Host{
        .{ .id = 0, .up = true, .ready = true, .has_model = false, .vram_limit = 8 * gb },
        .{ .id = 1, .up = true, .ready = true, .has_model = false, .vram_limit = 24 * gb },
    };
    try testing.expectEqual(Verdict{ .send_model = 1 }, place(&hosts, j));
    // A host with no image engine YET is still a candidate to send to: it has
    // none because it could not resolve a checkpoint.
    var bare = [_]Host{.{ .id = 3, .up = true, .ready = false, .has_model = false, .vram_limit = 24 * gb }};
    try testing.expectEqual(Verdict{ .send_model = 3 }, place(&bare, j));
    // A small card is still worth sending to: it will stream the model.
    var tiny = [_]Host{.{ .id = 0, .up = true, .ready = true, .has_model = false, .vram_limit = gb }};
    try testing.expectEqual(Verdict{ .send_model = 0 }, place(&tiny, j));
    // Nowhere only when nothing is up at all.
    var down = [_]Host{.{ .id = 0, .up = false, .ready = true, .has_model = false }};
    try testing.expectEqual(Verdict.nowhere, place(&down, j));
    // One host holding it beats any number that do not.
    var mixed = [_]Host{
        .{ .id = 0, .up = true, .ready = true, .has_model = false, .vram_limit = 24 * gb },
        .{ .id = 1, .up = true, .ready = true, .has_model = true, .vram_limit = 24 * gb, .queued = 5 },
    };
    try testing.expectEqual(Verdict{ .place = 1 }, place(&mixed, j));
}

test "a host's own finished images replace the assumed figures" {
    var t: Table = .{};
    const s: f64 = std.time.ns_per_s;
    // 20 steps at 1 mpx: 19 spans of 2 s, after a 30 s load. A real image is
    // always stamped with a real clock; `start_ns` 0 means "not stamped yet".
    try testing.expect(t.note(.{
        .family = "krea2",
        .mpx = 1,
        .steps = 20,
        .start_ns = @intFromFloat(10 * s),
        .first_step_ns = @intFromFloat(40 * s),
        .last_step_ns = @intFromFloat((40 + 38) * s),
    }));
    try testing.expectApproxEqAbs(@as(f64, 2), t.stepCost("krea2", 1), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 30), t.loadCost("krea2"), 1e-9);
    // Four times the area, four times the step, from the same measurement.
    try testing.expectApproxEqAbs(@as(f64, 8), t.stepCost("krea2", 4), 1e-9);
    // Another family is its own row, and an unseen one says nothing.
    try testing.expectEqual(@as(f64, 0), t.stepCost("sd15", 1));
    try testing.expectEqual(@as(f64, 0), t.loadCost("sd15"));

    // A second image on a resident pipeline moves the step figure and leaves
    // the load figure alone: two seconds before step one was not a load.
    try testing.expect(t.note(.{
        .family = "krea2",
        .mpx = 1,
        .steps = 20,
        .start_ns = @intFromFloat(100 * s),
        .first_step_ns = @intFromFloat(102 * s),
        .last_step_ns = @intFromFloat((102 + 19) * s),
    }));
    try testing.expectApproxEqAbs(@as(f64, 1.5), t.stepCost("krea2", 1), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 30), t.loadCost("krea2"), 1e-9);

    // What says nothing is refused: a cancelled image, one step, no timings.
    try testing.expect(!t.note(.{ .family = "krea2", .mpx = 1, .steps = 1, .first_step_ns = 1, .last_step_ns = 2 }));
    try testing.expect(!t.note(.{ .family = "", .mpx = 1, .steps = 20, .first_step_ns = 1, .last_step_ns = 2 }));
    try testing.expect(!t.note(.{ .family = "krea2", .mpx = 0, .steps = 20, .first_step_ns = 1, .last_step_ns = 2 }));
    try testing.expect(!t.note(.{ .family = "krea2", .mpx = 1, .steps = 20, .first_step_ns = 5, .last_step_ns = 5 }));
    try testing.expectApproxEqAbs(@as(f64, 1.5), t.stepCost("krea2", 1), 1e-9);
}

test "the measured table fills up without overwriting what it has" {
    var t: Table = .{};
    var buf: [8]u8 = undefined;
    for (0..max_families + 3) |i| {
        const fam = std.fmt.bufPrint(&buf, "fam{d}", .{i}) catch unreachable;
        _ = t.note(.{ .family = fam, .mpx = 1, .steps = 3, .first_step_ns = 1, .last_step_ns = 1 + 2 * std.time.ns_per_s });
    }
    try testing.expectEqual(max_families, t.n);
    try testing.expect(t.stepCost("fam0", 1) > 0);
    try testing.expectEqual(@as(f64, 0), t.stepCost("fam13", 1));
}
