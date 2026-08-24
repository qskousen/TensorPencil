//! Weight noise: a multiplicative jitter on the block-quant scales, injected into
//! the GEMM/GEMV kernels at the point where they widen `d` (and `dmin`) out of a
//! packed super-block.
//!
//! One draw covers a whole super-block, which is what makes this a perturbation of
//! the MODEL and not of the logits. Temperature can only reweight the distribution
//! the weights already produced; scaling a block's magnitude moves the function, so
//! the token ranking itself can change while staying locally coherent. It is also
//! the shape quantization error actually has, which makes a sigma sweep a direct
//! read on how much scale precision an architecture can lose before it stops
//! answering.
//!
//! The draw is keyed on the block's byte OFFSET WITHIN ITS WEIGHT, mixed with a
//! per-launch key the host folds from (seed, forward index, weight id). So every
//! block of every weight draws its own value, all lanes sharing a block agree on
//! it, one forward sees one coherent perturbed model, and the next forward draws
//! again. Nothing is written back: the resident weight is never touched, so a
//! streamed or offloaded layer cannot come back carrying different noise than it
//! left with.
//!
//! The offset rather than the absolute device address is what makes a run
//! reproducible: `cuMemAlloc` hands out different addresses every process, so an
//! address-keyed sweep produces a fresh unrepeatable model at each sigma and two
//! sigmas disagreeing tells you nothing. The offset is computed in 32 bits, which
//! is exact for any weight under 4 GiB (the largest here is a ~1.1 GB tied LM
//! head).
//!
//! sigma 0 leaves `d * 1.0`, exactly, so "off" is bit-identical and needs no branch.
//! That is what `mmq/gemv weight noise off is bit-identical` in backend.zig tests.
//!
//! ⚠️ Noise on `attn_k`/`attn_v` lands in the KV cache and persists for the rest of
//! the sequence, so those two drift cumulatively while every other weight is
//! resampled per forward. That is a different experiment from per-token resampling,
//! not a bug, but it is the reason a long reply keeps drifting after sigma goes back
//! to 0.

const std = @import("std");
const ptx = @import("ptx.zig");

/// Which of a block's two scales a draw is for. The salt is all that separates
/// them, so `d` and `dmin` of one block get independent jitter.
pub const Draw = enum(u32) {
    d = 0x00000000,
    dmin = 0x9E3779B9,
};

/// Loads sigma and the stream index into `sig`/`seq`, once per thread. Emit in the
/// kernel prologue; `jitter` then costs no parameter traffic in the inner loop.
pub const prologue_fmt =
    \\ld.param.f32 {[sig]s},[{[sig_param]s}];
    \\  ld.param.u32 {[key]s},[{[key_param]s}];
;

/// Multiplies `dst` by `1 + sigma*u`, with u in [-1,1] hashed from the super-block
/// address in `addr`, the stream index, and the draw's salt.
///
/// `h` and `t` are b32 scratch, `u` is f32 scratch, all clobbered. Three
/// multiply-xorshift rounds: the input differs between adjacent blocks only in a
/// few low address bits, and one round leaves those correlated across the whole
/// row, which shows up as a visible stripe pattern in the perturbation rather than
/// as noise.
pub const jitter_fmt =
    \\cvt.u32.u64 {[h]s},{[addr]s}; cvt.u32.u64 {[t]s},{[wbase]s}; sub.u32 {[h]s},{[h]s},{[t]s};
    \\  xor.b32 {[h]s},{[h]s},{[key]s}; xor.b32 {[h]s},{[h]s},0x{[salt]x};
    \\  mul.lo.u32 {[h]s},{[h]s},0x9E3779B1; shr.u32 {[t]s},{[h]s},15; xor.b32 {[h]s},{[h]s},{[t]s};
    \\  mul.lo.u32 {[h]s},{[h]s},0x85EBCA6B; shr.u32 {[t]s},{[h]s},13; xor.b32 {[h]s},{[h]s},{[t]s};
    \\  mul.lo.u32 {[h]s},{[h]s},0xC2B2AE35; shr.u32 {[t]s},{[h]s},16; xor.b32 {[h]s},{[h]s},{[t]s};
    \\  cvt.rn.f32.s32 {[u]s},{[h]s}; mul.f32 {[u]s},{[u]s},0f30000000;
    \\  fma.rn.f32 {[u]s},{[u]s},{[sig]s},0f3F800000; mul.f32 {[dst]s},{[dst]s},{[u]s};
;

/// Doc for the two comptime wrappers below plus the runtime one: `addr` is the
/// super-block's base address, `wbase` the weight's, and their difference is the
/// key's stable part.
///
/// `jitter_fmt` for the hand-authored kernels in elt.zig, which name their own
/// registers. Callers must widen the `.reg` declarations to cover the scratch.
pub fn jitterAt(
    comptime dst: []const u8,
    comptime addr: []const u8,
    comptime wbase: []const u8,
    comptime draw: Draw,
    comptime h: []const u8,
    comptime t: []const u8,
    comptime u: []const u8,
    comptime sig: []const u8,
    comptime key: []const u8,
) []const u8 {
    return "  " ++ std.fmt.comptimePrint(jitter_fmt, .{
        .dst = dst,
        .addr = addr,
        .wbase = wbase,
        .salt = @intFromEnum(draw),
        .h = h,
        .t = t,
        .u = u,
        .sig = sig,
        .key = key,
    }) ++ "\n";
}

/// `prologue_fmt` for the hand-authored kernels.
pub fn prologueAt(
    comptime sig: []const u8,
    comptime key: []const u8,
    comptime sig_param: []const u8,
    comptime key_param: []const u8,
) []const u8 {
    return "  " ++ std.fmt.comptimePrint(prologue_fmt, .{
        .sig = sig,
        .key = key,
        .sig_param = sig_param,
        .key_param = key_param,
    }) ++ "\n";
}

/// The scratch a `Builder` kernel needs for `emitJitter`, allocated once and
/// reused across draws so the register file does not grow per call site.
pub const Scratch = struct {
    sig: []const u8,
    key: []const u8,
    h: []const u8,
    t: []const u8,
    u: []const u8,

    /// Allocates the scratch and emits the prologue loads.
    pub fn init(b: *ptx.Builder, sig_param: []const u8, key_param: []const u8) !Scratch {
        const s: Scratch = .{
            .sig = try b.reg(.f32),
            .key = try b.reg(.b32),
            .h = try b.reg(.b32),
            .t = try b.reg(.b32),
            .u = try b.reg(.f32),
        };
        try b.linef(prologue_fmt, .{
            .sig = s.sig,
            .key = s.key,
            .sig_param = sig_param,
            .key_param = key_param,
        });
        return s;
    }
};

/// Emit a jitter of `dst`, keyed on the block at `addr` inside the weight at `wbase`.
pub fn emitJitter(b: *ptx.Builder, s: Scratch, dst: []const u8, addr: []const u8, wbase: []const u8, draw: Draw) !void {
    try b.linef(jitter_fmt, .{
        .dst = dst,
        .addr = addr,
        .wbase = wbase,
        .salt = @intFromEnum(draw),
        .h = s.h,
        .t = s.t,
        .u = s.u,
        .sig = s.sig,
        .key = s.key,
    });
}

/// The hash the kernels compute, for tests: the jitter applied to the block at
/// byte `offset` within its weight, under the launch `key`.
pub fn factor(offset: u32, key: u32, draw: Draw, sigma: f32) f32 {
    var h: u32 = offset;
    h ^= key;
    h ^= @intFromEnum(draw);
    h *%= 0x9E3779B1;
    h ^= h >> 15;
    h *%= 0x85EBCA6B;
    h ^= h >> 13;
    h *%= 0xC2B2AE35;
    h ^= h >> 16;
    const u = @as(f32, @floatFromInt(@as(i32, @bitCast(h)))) * 0x1p-31;
    return @mulAdd(f32, u, sigma, 1.0); // the kernel's fma, not a mul then an add
}

test "sigma 0 is exactly 1.0" {
    for ([_]u32{ 0, 1, 144, 0x7fff_ffff, 0xdead_beef }) |off| {
        try std.testing.expectEqual(@as(f32, 1.0), factor(off, 0, .d, 0));
        try std.testing.expectEqual(@as(f32, 1.0), factor(off, 12345, .dmin, 0));
    }
}

test "jitter stays inside 1 +- sigma and centers near 1" {
    const sigma: f32 = 0.05;
    var sum: f64 = 0;
    const n = 4096;
    for (0..n) |i| {
        // Successive q4_k super-blocks of one row: 144 bytes apart.
        const f = factor(@intCast(i * 144), 7, .d, sigma);
        errdefer std.debug.print("i={d} f={d}\n", .{ i, f });
        try std.testing.expect(f >= 1.0 - sigma and f <= 1.0 + sigma);
        sum += f;
    }
    const mean = sum / n;
    errdefer std.debug.print("mean={d}\n", .{mean});
    try std.testing.expect(@abs(mean - 1.0) < sigma * 0.05);
}

test "d and dmin draw independently, and every forward redraws" {
    const s: f32 = 0.1;
    const off: u32 = 0x1234_5000;
    try std.testing.expect(factor(off, 3, .d, s) != factor(off, 3, .dmin, s));
    try std.testing.expect(factor(off, 3, .d, s) != factor(off, 4, .d, s));
    // Adjacent blocks of a row must not share a draw.
    try std.testing.expect(factor(off, 3, .d, s) != factor(off + 144, 3, .d, s));
}
