//! The activation-precision dial: per component (text encoder, denoiser, VAE) an
//! f32 / f16 / bf16 choice, or `auto`.
//!
//! `auto` starts at the family's default and, when a forward comes out non-finite,
//! widens to f32, never to bf16: bf16 buys f16's overflow its range back at three
//! mantissa bits, where f32 costs speed only on the forward that needed it. bf16 runs
//! when it is asked for. A dtype the component would run as another is skipped. The widened choice is REMEMBERED
//! for the session, so later images do not retry; it clears when the dial changes and
//! with the session. An explicit dtype gets the same check and no fallback: it fails.
//! No calibration forward, nothing written next to a checkpoint.

const std = @import("std");

pub const Act = @import("tp_gpu").context.Dt;

pub const Component = enum {
    encoder,
    denoiser,
    vae,

    pub fn label(self: Component) []const u8 {
        return switch (self) {
            .encoder => "text encoder",
            .denoiser => "denoiser",
            .vae => "VAE",
        };
    }
};

/// Widening order. bf16 is reached only by asking for it, and widens to f32.
const ladder = [_]Act{ .f16, .f32 };

fn rank(a: Act) usize {
    return if (a == .bf16) 0 else std.mem.indexOfScalar(Act, &ladder, a).?;
}

/// One component's dial and what `auto` has learned this session.
pub const State = struct {
    /// Null is `auto`.
    dial: ?Act = null,
    /// The dtype `auto` widened to after a non-finite forward.
    fallback: ?Act = null,

    /// Take this render's dial; a different one drops what `auto` learned.
    pub fn setDial(self: *State, dial: ?Act) void {
        if (!std.meta.eql(self.dial, dial)) self.fallback = null;
        self.dial = dial;
    }

    /// The dtype to run, given the family's default.
    pub fn current(self: State, default: Act) Act {
        return self.dial orelse self.fallback orelse default;
    }

    /// After a non-finite forward at `ran`: the next wider dtype `runs` accepts, now
    /// remembered, or null when there is none or the dial is explicit.
    pub fn widen(self: *State, ran: Act, ctx: anytype, comptime runs: fn (@TypeOf(ctx), Act) bool) ?Act {
        const a = self.next(ran, ctx, runs) orelse return null;
        self.fallback = a;
        return a;
    }

    /// What `widen` would choose, without choosing it.
    pub fn next(self: State, ran: Act, ctx: anytype, comptime runs: fn (@TypeOf(ctx), Act) bool) ?Act {
        if (self.dial != null) return null;
        for (ladder[rank(ran) + 1 ..]) |a| if (runs(ctx, a)) return a;
        return null;
    }
};

/// How a component's integer convrot GEMMs ran: the weight's bits and what the
/// activation met them as. `W8A8` is int8 convrot's own arithmetic, `W8A16` / `W8A32`
/// the weight-only routes an explicit dial selects.
pub const IntArith = struct {
    w_bits: u4,
    act: enum { int8, int4, f16, bf16, f32 },

    pub fn format(self: IntArith, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("W{d}A{s}", .{ self.w_bits, switch (self.act) {
            .int8 => "8",
            .int4 => "4",
            .f16 => "16",
            .bf16 => "16 bf16",
            .f32 => "32",
        } });
    }
};

/// What each component ran, for the saved PNG's `Activation dtype` field: the
/// activation dtype, and for a component with integer weights the GEMM arithmetic.
pub const Acts = struct {
    encoder: Act,
    denoiser: Act,
    vae: Act,
    encoder_int: ?IntArith = null,
    denoiser_int: ?IntArith = null,

    pub fn format(self: Acts, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("text encoder {t}", .{self.encoder});
        if (self.encoder_int) |i| try w.print(" ({f})", .{i});
        try w.print(", denoiser {t}", .{self.denoiser});
        if (self.denoiser_int) |i| try w.print(" ({f})", .{i});
        try w.print(", VAE {t}", .{self.vae});
    }
};

/// What failed, for the error a caller reports: `component` produced a non-finite
/// output at `act`, at sampling step `step` when there is one.
pub const Failure = struct {
    component: Component,
    act: Act,
    step: ?usize = null,

    pub fn format(self: Failure, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("{s} produced a non-finite output at {t}", .{ self.component.label(), self.act });
        if (self.step) |s| try w.print(" (step {d})", .{s + 1});
    }
};

/// DIAGNOSTIC: count `component`'s output at `act` as non-finite, so the fallback can
/// be exercised on a real render (`--inject-nonfinite`). Never set in a normal run.
pub var inject: ?struct { component: Component, act: Act } = null;

pub fn injected(component: Component, act: Act) bool {
    const i = inject orelse return false;
    return i.component == component and i.act == act;
}

pub fn allFinite(xs: []const f32) bool {
    for (xs) |x| if (!std.math.isFinite(x)) return false;
    return true;
}

test "auto widens over what runs, remembers it, and forgets it with the dial" {
    const Runs = struct {
        fn f(ok: []const Act, a: Act) bool {
            return std.mem.indexOfScalar(Act, ok, a) != null;
        }
    };
    var s: State = .{};
    try std.testing.expectEqual(Act.f16, s.current(.f16));
    // Never through bf16, even where the component runs it.
    try std.testing.expectEqual(@as(?Act, .f32), s.widen(.f16, @as([]const Act, &.{ .f16, .bf16, .f32 }), Runs.f));
    try std.testing.expectEqual(Act.f32, s.current(.f16));
    try std.testing.expectEqual(@as(?Act, null), s.widen(.f32, @as([]const Act, &.{ .f16, .bf16, .f32 }), Runs.f));
    s.setDial(null);
    try std.testing.expectEqual(Act.f32, s.current(.f16));
    s.setDial(.bf16);
    try std.testing.expectEqual(Act.bf16, s.current(.f16));
    // An explicit dial never widens.
    try std.testing.expectEqual(@as(?Act, null), s.widen(.bf16, @as([]const Act, &.{ .f16, .bf16, .f32 }), Runs.f));
    s.setDial(null);
    try std.testing.expectEqual(Act.f16, s.current(.f16));
}

test "Acts reads as the PNG field writes it" {
    var buf: [96]u8 = undefined;
    const a: Acts = .{ .encoder = .f32, .denoiser = .bf16, .vae = .f16 };
    try std.testing.expectEqualStrings("text encoder f32, denoiser bf16, VAE f16", try std.fmt.bufPrint(&buf, "{f}", .{a}));
    const b: Acts = .{ .encoder = .f32, .denoiser = .f16, .vae = .f32, .encoder_int = .{ .w_bits = 8, .act = .int8 }, .denoiser_int = .{ .w_bits = 4, .act = .f32 } };
    try std.testing.expectEqualStrings("text encoder f32 (W8A8), denoiser f16 (W4A32), VAE f32", try std.fmt.bufPrint(&buf, "{f}", .{b}));
}

test "allFinite" {
    try std.testing.expect(allFinite(&.{ 1, -2, 0 }));
    try std.testing.expect(!allFinite(&.{ 1, std.math.nan(f32) }));
    try std.testing.expect(!allFinite(&.{std.math.inf(f32)}));
}
