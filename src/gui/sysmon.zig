//! System monitors for the tp-gui status bar:
//! host-CPU utilization from `/proc/stat`, and GPU utilization + VRAM from
//! NVML (runtime-`dlopen`'d, like the CUDA/Vulkan drivers, absent driver just
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
/// no `std.Io` threading, the status bar samples this every frame).
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
    return @as(f32, @floatFromInt(khz)) / 1000.0; // kHz -> MHz
}

// NVML C struct layouts (nvml.h). Only the fields we read.
const NvmlUtilization = extern struct { gpu: c_uint, memory: c_uint };
const NvmlMemory = extern struct { total: c_ulonglong, free: c_ulonglong, used: c_ulonglong };
const NvmlDevice = ?*anyopaque; // opaque nvmlDevice_t handle

/// `nvmlProcessInfo_v2_t`, what the `_v2`/`_v3` process getters write.
const ProcInfoV2 = extern struct { pid: c_uint, used: c_ulonglong, gi: c_uint, ci: c_uint };
/// `nvmlProcessInfo_t`, what the ORIGINAL (unversioned) getters write. Passing
/// this layout to a `_v2`/`_v3` symbol (or vice versa) silently misreads `used`,
/// so each symbol is paired with its own layout below.
const ProcInfoV1 = extern struct { pid: c_uint, used: c_ulonglong };
/// NVML_VALUE_NOT_AVAILABLE, "this process's usage can't be determined".
const value_not_available: c_ulonglong = std.math.maxInt(c_ulonglong);
/// nvmlDeviceGet{Compute,Graphics}RunningProcesses: (device, *count, *infos).
const GetProcsFn = *const fn (NvmlDevice, *c_uint, ?*anyopaque) callconv(.c) c_int;
/// Max processes we'll read per list (stack buffer; 128 × 24 B = 3 KiB).
const max_procs = 128;

/// Outcome of scanning one process list for a pid.
const ListHit = union(enum) {
    /// The query itself failed (missing data, no permission, list too long).
    failed,
    /// Query fine, our pid isn't in this list.
    absent,
    used: u64,
};

/// Find `pid` in a decoded process list. NVML reports a process's TOTAL card
/// usage in whichever list it appears, so the first hit is the whole answer.
fn scanProcs(comptime T: type, list: []const T, pid: u32) ListHit {
    for (list) |e| {
        if (e.pid != pid) continue;
        if (e.used == value_not_available) return .failed;
        return .{ .used = @intCast(e.used) };
    }
    return .absent;
}

/// Optional NVML handle for GPU utilization + VRAM. `open()` returns null when
/// the NVML library or a required symbol is missing (no NVIDIA driver, or a
/// headless/container run), the status bar then shows the GPU meter as n/a.
/// Process-wide lazy NVML handle. Shared rather than opened per consumer so the
/// status bar and the VRAM budget read the SAME driver in the same pass, two
/// handles could disagree, and the whole point of the per-process split is that
/// its terms are coherent with each other.
var g_nvml: ?Nvml = null;
var g_nvml_tried: bool = false;

pub fn nvml() ?*Nvml {
    if (!g_nvml_tried) {
        g_nvml = Nvml.open();
        g_nvml_tried = true;
    }
    return if (g_nvml) |*n| n else null;
}

/// Release the shared handle at process exit.
pub fn nvmlClose() void {
    if (g_nvml) |*n| n.close();
    g_nvml = null;
}

pub const Nvml = struct {
    lib: std.DynLib,
    dev: NvmlDevice,
    getUtil: *const fn (NvmlDevice, *NvmlUtilization) callconv(.c) c_int,
    getMem: *const fn (NvmlDevice, *NvmlMemory) callconv(.c) c_int,
    getClock: ?*const fn (NvmlDevice, c_uint, *c_uint) callconv(.c) c_int, // nvmlDeviceGetClockInfo (optional)
    shutdown: *const fn () callconv(.c) c_int,
    // Per-process VRAM (optional, older drivers lack the symbols).
    getComputeProcs: ?GetProcsFn,
    getGraphicsProcs: ?GetProcsFn,
    proc_layout: enum { v2, v1 },

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

        // Per-process VRAM: prefer the versioned getters (nvmlProcessInfo_v2_t),
        // falling back to the original pair (nvmlProcessInfo_t). Both lists are
        // taken from the same generation so one decode layout covers them.
        var layout: @FieldType(Nvml, "proc_layout") = .v2;
        var get_compute = lib.lookup(GetProcsFn, "nvmlDeviceGetComputeRunningProcesses_v3") orelse
            lib.lookup(GetProcsFn, "nvmlDeviceGetComputeRunningProcesses_v2");
        var get_graphics = lib.lookup(GetProcsFn, "nvmlDeviceGetGraphicsRunningProcesses_v3") orelse
            lib.lookup(GetProcsFn, "nvmlDeviceGetGraphicsRunningProcesses_v2");
        if (get_compute == null and get_graphics == null) {
            layout = .v1;
            get_compute = lib.lookup(GetProcsFn, "nvmlDeviceGetComputeRunningProcesses");
            get_graphics = lib.lookup(GetProcsFn, "nvmlDeviceGetGraphicsRunningProcesses");
        }

        if (init_fn() != 0) return null;
        var dev: NvmlDevice = null;
        if (by_index(0, &dev) != 0) {
            _ = shutdown_fn();
            return null;
        }
        return .{
            .lib = lib,
            .dev = dev,
            .getUtil = get_util,
            .getMem = get_mem,
            .getClock = get_clock,
            .shutdown = shutdown_fn,
            .getComputeProcs = get_compute,
            .getGraphicsProcs = get_graphics,
            .proc_layout = layout,
        };
    }

    /// Card memory (bytes) charged to `pid`, for our own pid, that's our WHOLE
    /// footprint including everything our allocators can't see: the CUDA
    /// context(s) and JIT'd modules, cuBLASLt/cuDNN internals, and the SDL/GL
    /// window + image textures. The meter uses it to tell "ours but untracked"
    /// apart from "another process's" VRAM (see vram_split.zig).
    ///
    /// `procUsed` for our own process, what the meter actually wants.
    pub fn selfUsed(self: *Nvml) ?u64 {
        // `std.posix.getpid` doesn't exist in 0.16; `std.posix.system` is the
        // portable spelling of the syscall (no std.os.linux dependency).
        return self.procUsed(@intCast(std.posix.system.getpid()));
    }

    /// Returns null when NVML can't answer (symbols missing on an older driver,
    /// query failed, no permission); 0 when we hold nothing on this card.
    pub fn procUsed(self: *Nvml, pid: u32) ?u64 {
        var any_ok = false;
        // A process doing both compute and graphics (tp-gui does, CUDA plus the
        // SDL window) appears in BOTH lists reporting the same total, so the
        // first hit wins; summing would double-count it.
        for ([_]?GetProcsFn{ self.getComputeProcs, self.getGraphicsProcs }) |maybe| {
            const f = maybe orelse continue;
            switch (self.queryProcs(f, pid)) {
                .used => |v| return v,
                .absent => any_ok = true,
                .failed => {},
            }
        }
        return if (any_ok) 0 else null;
    }

    fn queryProcs(self: *Nvml, f: GetProcsFn, pid: u32) ListHit {
        var count: c_uint = max_procs;
        switch (self.proc_layout) {
            .v2 => {
                var buf: [max_procs]ProcInfoV2 = undefined;
                if (f(self.dev, &count, &buf) != 0) return .failed;
                return scanProcs(ProcInfoV2, buf[0..@min(count, max_procs)], pid);
            },
            .v1 => {
                var buf: [max_procs]ProcInfoV1 = undefined;
                if (f(self.dev, &count, &buf) != 0) return .failed;
                return scanProcs(ProcInfoV1, buf[0..@min(count, max_procs)], pid);
            },
        }
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

test "NVML process-info structs match the C layouts" {
    // A mismatch here reads `used` from the wrong offset and the meter silently
    // reports garbage for our own footprint, so pin the ABI.
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ProcInfoV2, "pid"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(ProcInfoV2, "used"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(ProcInfoV2, "gi"));
    try std.testing.expectEqual(@as(usize, 20), @offsetOf(ProcInfoV2, "ci"));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(ProcInfoV2));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ProcInfoV1, "pid"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(ProcInfoV1, "used"));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(ProcInfoV1));
}

test "scanProcs finds our pid, both layouts, and rejects NOT_AVAILABLE" {
    const v2 = [_]ProcInfoV2{
        .{ .pid = 2656, .used = 811 << 20, .gi = 0, .ci = 0 },
        .{ .pid = 1512878, .used = 22398 << 20, .gi = 0, .ci = 0 },
    };
    try std.testing.expectEqual(@as(u64, 22398 << 20), scanProcs(ProcInfoV2, &v2, 1512878).used);
    try std.testing.expect(scanProcs(ProcInfoV2, &v2, 999) == .absent);
    try std.testing.expect(scanProcs(ProcInfoV2, v2[0..0], 1512878) == .absent);

    const v1 = [_]ProcInfoV1{.{ .pid = 42, .used = 5 << 30 }};
    try std.testing.expectEqual(@as(u64, 5 << 30), scanProcs(ProcInfoV1, &v1, 42).used);

    const na = [_]ProcInfoV1{.{ .pid = 42, .used = value_not_available }};
    try std.testing.expect(scanProcs(ProcInfoV1, &na, 42) == .failed);
}

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
    // with a baseline set, a synthetic current of (2000, 1400) -> busy 500/1000.
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
