//! Lightweight wall-time profiler for the CPU forward path. `perf` isn't
//! available on this kernel, so we accumulate ns per category across a run and
//! print a breakdown under `--profile` (cpu backend). Regions must NOT overlap
//! (matmul self-times; the layer code times only its non-matmul sections), so
//! the buckets sum to a meaningful total. No-op and ~free when disabled.

const std = @import("std");
const builtin = @import("builtin");

pub const Cat = enum { matmul, deltanet, conv, attention, rope, norm, act, embed, other };
const n_cat = @typeInfo(Cat).@"enum".fields.len;

pub var enabled: bool = false;
var acc: [n_cat]u64 = [_]u64{0} ** n_cat;
var calls: [n_cat]u64 = [_]u64{0} ** n_cat;

pub inline fn tic() u64 {
    return if (enabled) nowNs() else 0;
}
pub inline fn toc(cat: Cat, t0: u64) void {
    if (!enabled) return;
    acc[@intFromEnum(cat)] += nowNs() -| t0;
    calls[@intFromEnum(cat)] += 1;
}
pub fn reset() void {
    acc = [_]u64{0} ** n_cat;
    calls = [_]u64{0} ** n_cat;
}
pub fn report(w: *std.Io.Writer) !void {
    var total: u64 = 0;
    for (acc) |a| total += a;
    if (total == 0) return;
    const tf: f64 = @floatFromInt(total);
    try w.print("\n[cpu profile — non-overlapping buckets]\n", .{});
    inline for (@typeInfo(Cat).@"enum".fields, 0..) |f, i| {
        if (acc[i] > 0) try w.print("  {s:<10} {d:>8.1} ms  {d:>5.1}%  ({d} calls)\n", .{
            f.name,
            @as(f64, @floatFromInt(acc[i])) / 1e6,
            100.0 * @as(f64, @floatFromInt(acc[i])) / tf,
            calls[i],
        });
    }
    try w.print("  {s:<10} {d:>8.1} ms\n", .{ "TOTAL", tf / 1e6 });
}

/// Monotonic nanoseconds, io-free (`std.time.Timer` is gone in 0.16 and
/// `std.Io.Clock` needs an `Io` the callers here do not have). Shared with the
/// CUDA backend and the benchmarks, which all time regions the same way.
pub fn monoNs() u64 {
    if (builtin.os.tag == .windows) {
        var freq: u64 = undefined;
        var counter: u64 = undefined;
        _ = std.os.windows.ntdll.RtlQueryPerformanceFrequency(@ptrCast(&freq));
        _ = std.os.windows.ntdll.RtlQueryPerformanceCounter(@ptrCast(&counter));
        if (freq == 0) return 0;
        // Scale before dividing, in two halves, so neither overflows at uptimes
        // a benchmark run can reach.
        return (counter / freq) * 1_000_000_000 + ((counter % freq) * 1_000_000_000) / freq;
    }
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const nowNs = monoNs;
