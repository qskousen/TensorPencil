//! Lightweight wall-time profiler for the CPU forward path. `perf` isn't
//! available on this kernel, so we accumulate ns per category across a run and
//! print a breakdown under `--profile` (cpu backend). Regions must NOT overlap
//! (matmul self-times; the layer code times only its non-matmul sections), so
//! the buckets sum to a meaningful total. No-op and ~free when disabled.

const std = @import("std");

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

fn nowNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}
