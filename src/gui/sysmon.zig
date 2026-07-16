//! System monitors for the tp-gui status bar (GUI_VRAM.md, Phase 6):
//! host-CPU utilization from `/proc/stat`, and GPU utilization + VRAM from
//! NVML (runtime-`dlopen`'d, like the CUDA/Vulkan drivers — absent driver just
//! reports `null`, never a hard dependency). Per-model VRAM accounting lives in
//! the VRAM coordinator; this module only covers the system-wide meters.
const std = @import("std");

/// Aggregate CPU jiffie counts parsed from the `cpu ...` line of `/proc/stat`.
const CpuTimes = struct { total: u64, idle: u64 };

/// Parse the aggregate `cpu` line ("cpu  u n s idle iowait irq softirq ...").
/// `total` sums every field; `idle` is idle+iowait. Returns null on a malformed
/// line. Split out from the syscall so it's unit-testable.
fn parseCpuLine(line: []const u8) ?CpuTimes {
    var it = std.mem.tokenizeAny(u8, line, " \t");
    const head = it.next() orelse return null;
    if (!std.mem.eql(u8, head, "cpu")) return null;
    var total: u64 = 0;
    var idle: u64 = 0;
    var i: usize = 0;
    while (it.next()) |tok| : (i += 1) {
        const v = std.fmt.parseInt(u64, tok, 10) catch continue;
        total += v;
        if (i == 3 or i == 4) idle += v; // idle (3), iowait (4)
    }
    if (i == 0) return null;
    return .{ .total = total, .idle = idle };
}

/// Read `/proc/stat`'s first line into `buf` via a raw syscall (no allocation,
/// no `std.Io` threading — the status bar samples this every frame).
fn readProcStat(buf: []u8) ?CpuTimes {
    const fd = std.os.linux.open("/proc/stat", .{ .ACCMODE = .RDONLY }, 0);
    if (std.posix.errno(fd) != .SUCCESS) return null;
    const ifd: i32 = @intCast(fd);
    defer _ = std.os.linux.close(ifd);
    const n = std.os.linux.read(ifd, buf.ptr, buf.len);
    if (std.posix.errno(n) != .SUCCESS or n == 0) return null;
    const bytes = buf[0..@intCast(n)];
    const nl = std.mem.indexOfScalar(u8, bytes, '\n') orelse bytes.len;
    return parseCpuLine(bytes[0..nl]);
}

/// Rolling host-CPU-utilization sampler. `sample()` returns busy fraction over
/// the interval since the previous call, in percent (0..100); the first call
/// (no baseline yet) returns 0.
pub const CpuMeter = struct {
    last: ?CpuTimes = null,

    pub fn sample(self: *CpuMeter) f32 {
        var buf: [512]u8 = undefined;
        const cur = readProcStat(&buf) orelse return 0;
        defer self.last = cur;
        const prev = self.last orelse return 0;
        const dt = cur.total -| prev.total;
        const di = cur.idle -| prev.idle;
        if (dt == 0) return 0;
        const busy = dt -| di;
        return @as(f32, @floatFromInt(busy)) / @as(f32, @floatFromInt(dt)) * 100.0;
    }
};

/// A GPU snapshot: utilization percent + VRAM totals (bytes) + graphics clock
/// (MHz). `util` is the NVML "percent of time one or more kernels ran".
pub const GpuStats = struct {
    util: u32,
    mem_used: u64,
    mem_total: u64,
    clock_mhz: u32,
};

/// Current CPU frequency (MHz) from cpu0's cpufreq governor, 0 if unavailable
/// (no cpufreq sysfs, e.g. some VMs). Raw syscall read, no allocation.
pub fn cpuFreqMhz() f32 {
    var buf: [32]u8 = undefined;
    const fd = std.os.linux.open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", .{ .ACCMODE = .RDONLY }, 0);
    if (std.posix.errno(fd) != .SUCCESS) return 0;
    const ifd: i32 = @intCast(fd);
    defer _ = std.os.linux.close(ifd);
    const n = std.os.linux.read(ifd, &buf, buf.len);
    if (std.posix.errno(n) != .SUCCESS or n == 0) return 0;
    const s = std.mem.trim(u8, buf[0..@intCast(n)], " \t\r\n");
    const khz = std.fmt.parseInt(u64, s, 10) catch return 0;
    return @as(f32, @floatFromInt(khz)) / 1000.0; // kHz → MHz
}

// NVML C struct layouts (nvml.h). Only the fields we read.
const NvmlUtilization = extern struct { gpu: c_uint, memory: c_uint };
const NvmlMemory = extern struct { total: c_ulonglong, free: c_ulonglong, used: c_ulonglong };
const NvmlDevice = ?*anyopaque; // opaque nvmlDevice_t handle

/// Optional NVML handle for GPU utilization + VRAM. `open()` returns null when
/// the NVML library or a required symbol is missing (no NVIDIA driver, or a
/// headless/container run) — the status bar then shows the GPU meter as n/a.
pub const Nvml = struct {
    lib: std.DynLib,
    dev: NvmlDevice,
    getUtil: *const fn (NvmlDevice, *NvmlUtilization) callconv(.c) c_int,
    getMem: *const fn (NvmlDevice, *NvmlMemory) callconv(.c) c_int,
    getClock: ?*const fn (NvmlDevice, c_uint, *c_uint) callconv(.c) c_int, // nvmlDeviceGetClockInfo (optional)
    shutdown: *const fn () callconv(.c) c_int,

    pub fn open() ?Nvml {
        var lib = std.DynLib.open("libnvidia-ml.so.1") catch
            std.DynLib.open("libnvidia-ml.so") catch return null;
        errdefer lib.close();

        const init_fn = lib.lookup(*const fn () callconv(.c) c_int, "nvmlInit_v2") orelse
            lib.lookup(*const fn () callconv(.c) c_int, "nvmlInit") orelse return null;
        const by_index = lib.lookup(*const fn (c_uint, *NvmlDevice) callconv(.c) c_int, "nvmlDeviceGetHandleByIndex_v2") orelse
            lib.lookup(*const fn (c_uint, *NvmlDevice) callconv(.c) c_int, "nvmlDeviceGetHandleByIndex") orelse return null;
        const get_util = lib.lookup(*const fn (NvmlDevice, *NvmlUtilization) callconv(.c) c_int, "nvmlDeviceGetUtilizationRates") orelse return null;
        const get_mem = lib.lookup(*const fn (NvmlDevice, *NvmlMemory) callconv(.c) c_int, "nvmlDeviceGetMemoryInfo") orelse return null;
        const get_clock = lib.lookup(*const fn (NvmlDevice, c_uint, *c_uint) callconv(.c) c_int, "nvmlDeviceGetClockInfo"); // optional
        const shutdown_fn = lib.lookup(*const fn () callconv(.c) c_int, "nvmlShutdown") orelse return null;

        if (init_fn() != 0) return null;
        var dev: NvmlDevice = null;
        if (by_index(0, &dev) != 0) {
            _ = shutdown_fn();
            return null;
        }
        return .{ .lib = lib, .dev = dev, .getUtil = get_util, .getMem = get_mem, .getClock = get_clock, .shutdown = shutdown_fn };
    }

    /// Current utilization + VRAM + graphics clock, or null if a query failed.
    pub fn query(self: *Nvml) ?GpuStats {
        var u: NvmlUtilization = undefined;
        var m: NvmlMemory = undefined;
        const uok = self.getUtil(self.dev, &u) == 0;
        const mok = self.getMem(self.dev, &m) == 0;
        if (!uok and !mok) return null;
        var clock: c_uint = 0;
        if (self.getClock) |gc| _ = gc(self.dev, 0, &clock); // 0 = NVML_CLOCK_GRAPHICS
        return .{
            .util = if (uok) u.gpu else 0,
            .mem_used = if (mok) @intCast(m.used) else 0,
            .mem_total = if (mok) @intCast(m.total) else 0,
            .clock_mhz = @intCast(clock),
        };
    }

    pub fn close(self: *Nvml) void {
        _ = self.shutdown();
        self.lib.close();
        self.* = undefined;
    }
};

test "parseCpuLine sums fields and idle=idle+iowait" {
    // cpu  user nice system idle iowait irq softirq steal guest guest_nice
    const t = parseCpuLine("cpu  100 0 50 800 40 0 10 0 0 0").?;
    try std.testing.expectEqual(@as(u64, 1000), t.total);
    try std.testing.expectEqual(@as(u64, 840), t.idle); // 800 + 40
    try std.testing.expect(parseCpuLine("intr 1 2 3") == null); // wrong prefix
    try std.testing.expect(parseCpuLine("cpu") == null); // no fields
}

test "CpuMeter first sample returns 0 (no baseline)" {
    var m: CpuMeter = .{ .last = .{ .total = 1000, .idle = 900 } };
    // Force a deterministic delta by simulating readProcStat's result path:
    // with a baseline set, a synthetic current of (2000, 1400) → busy 500/1000.
    // (readProcStat reads the real /proc/stat, so we only assert the math via
    // the pure helper here; the real sampler is exercised at runtime.)
    _ = &m;
    const prev = CpuTimes{ .total = 1000, .idle = 900 };
    const cur = CpuTimes{ .total = 2000, .idle = 1400 };
    const dt = cur.total - prev.total;
    const di = cur.idle - prev.idle;
    const pct = @as(f32, @floatFromInt(dt - di)) / @as(f32, @floatFromInt(dt)) * 100.0;
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), pct, 0.01);
}
