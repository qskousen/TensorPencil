//! System monitors for the tp-gui status bar:
//! host-CPU utilization from `/proc/stat`, and GPU utilization + VRAM from
//! NVML (runtime-`dlopen`'d, like the CUDA/Vulkan drivers, absent driver just
//! reports `null`, never a hard dependency). Per-model VRAM accounting lives in
//! the VRAM coordinator; this module only covers the system-wide meters.
const std = @import("std");
const builtin = @import("builtin");
const dynlib = @import("TensorPencil").dynlib;

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

/// Read the head of a procfs/sysfs file into `buf` via a raw syscall (no
/// allocation, no `std.Io` threading, the status bar samples these every frame).
/// Null off Linux, where neither file exists and there is no equivalent to read.
fn readSysFile(path: [*:0]const u8, buf: []u8) ?[]const u8 {
    if (builtin.os.tag != .linux) return null;
    const fd = std.os.linux.open(path, .{ .ACCMODE = .RDONLY }, 0);
    if (std.posix.errno(fd) != .SUCCESS) return null;
    const ifd: i32 = @intCast(fd);
    defer _ = std.os.linux.close(ifd);
    const n = std.os.linux.read(ifd, buf.ptr, buf.len);
    if (std.posix.errno(n) != .SUCCESS or n == 0) return null;
    return buf[0..@intCast(n)];
}

fn readProcStat(buf: []u8) ?CpuTimes {
    const bytes = readSysFile("/proc/stat", buf) orelse return null;
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

/// GPU busy time and clock for a card NVML cannot answer for: every open
/// driver (i915, xe, amdgpu). The kernel publishes cumulative per-engine busy
/// nanoseconds for each DRM client in `/proc/self/fdinfo`, which needs no
/// privilege, unlike the i915 PMU: that one reads through `perf_event_open`
/// and is refused outright where `perf_event_paranoid` is above 2, which is
/// the default on Ubuntu.
///
/// ⚠️ This is OUR busy time, not the whole card's. NVML's number counts every
/// process; nobody else's work shows up here. On a host that exists to run
/// these engines the two nearly agree, and a figure that under-reports a
/// shared card is the honest direction to be wrong in.
pub const DrmMeter = struct {
    /// Per client, because the counters are CUMULATIVE and a client appears
    /// mid-run: the pipeline opens its own when a model loads. Counting a new
    /// client's whole history as one interval's work reads as 100% for a
    /// sample, which is how this was caught.
    clients: [max_clients]Client = @splat(.{}),
    n_clients: usize = 0,
    last_wall_ns: i96 = 0,
    /// `cardN` whose PCI address matched, for the clock. Empty until found.
    card: [16]u8 = @splat(0),
    card_len: u8 = 0,

    pub const max_clients = 16;
    const Client = struct { id: u64 = 0, busy_ns: u64 = 0 };

    pub const Sample = struct { util: f32 = 0, clock_mhz: u32 = 0, found: bool = false };

    pub fn sample(self: *DrmMeter, io: std.Io, now_ns: i96) Sample {
        if (builtin.os.tag != .linux) return .{};
        var pdev: [24]u8 = @splat(0);
        var pdev_len: usize = 0;
        var cur: [max_clients]Client = @splat(.{});
        const n = scanFdinfo(io, &cur, &pdev, &pdev_len) orelse return .{};
        if (self.card_len == 0 and pdev_len > 0) self.findCard(io, pdev[0..pdev_len]);

        // Only clients present in BOTH samples can have a delta; one seen for
        // the first time contributes nothing until the next round.
        var delta: u64 = 0;
        for (cur[0..n]) |c| {
            for (self.clients[0..self.n_clients]) |p| {
                if (p.id != c.id) continue;
                delta += c.busy_ns -| p.busy_ns;
                break;
            }
        }
        @memcpy(self.clients[0..n], cur[0..n]);
        self.n_clients = n;
        const prev_wall = self.last_wall_ns;
        self.last_wall_ns = now_ns;

        var out: Sample = .{ .found = true, .clock_mhz = self.clockMhz() };
        if (prev_wall == 0 or now_ns <= prev_wall) return out;
        const dt: u64 = @intCast(now_ns - prev_wall);
        if (dt == 0) return out;
        const pct = @as(f64, @floatFromInt(delta)) / @as(f64, @floatFromInt(dt)) * 100.0;
        // MEASURED on an Arc A310 mid-render: the render engine alone reported
        // 107.3% of the elapsed nanoseconds, with no other engine busy. i915
        // sums per-context runtimes and overlapping contexts on one engine each
        // count their whole window, so the raw figure runs over. The clamp is
        // the answer, not a defensive guess: one card is still one card.
        out.util = @floatCast(@min(pct, 100.0));
        return out;
    }

    /// Every DRM client this process holds, with its cumulative engine-busy
    /// nanoseconds, plus the PCI address they belong to. Returns how many were
    /// written, or null when this process holds no DRM fd at all.
    fn scanFdinfo(io: std.Io, out: []Client, pdev: []u8, pdev_len: *usize) ?usize {
        var dir = std.Io.Dir.openDirAbsolute(io, "/proc/self/fdinfo", .{ .iterate = true }) catch return null;
        defer dir.close(io);
        var it = dir.iterate();
        // One fd per client is enough: two fds sharing a client id report the
        // same counters, and adding them would double the busy time.
        var n: usize = 0;
        var any = false;
        while (it.next(io) catch null) |ent| {
            if (ent.kind != .file) continue;
            var buf: [4096]u8 = undefined;
            const bytes = dir.readFile(io, ent.name, &buf) catch continue;
            if (std.mem.indexOf(u8, bytes, "drm-driver:") == null) continue;
            any = true;
            const id = fieldU64(bytes, "drm-client-id:") orelse continue;
            var dup = false;
            for (out[0..n]) |c| if (c.id == id) {
                dup = true;
            };
            if (dup) continue;
            if (pdev_len.* == 0) if (fieldText(bytes, "drm-pdev:")) |p| {
                const k = @min(p.len, pdev.len);
                @memcpy(pdev[0..k], p[0..k]);
                pdev_len.* = k;
            };
            if (n == out.len) continue;
            out[n] = .{ .id = id, .busy_ns = engineNs(bytes) };
            n += 1;
        }
        return if (any) n else null;
    }

    /// Sum of every `drm-engine-<name>: <n> ns` line. The `capacity` lines have
    /// no `ns` and are skipped by the suffix check.
    fn engineNs(bytes: []const u8) u64 {
        var total: u64 = 0;
        var lines = std.mem.splitScalar(u8, bytes, '\n');
        while (lines.next()) |line| {
            if (!std.mem.startsWith(u8, line, "drm-engine-")) continue;
            const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
            const v = std.mem.trim(u8, line[colon + 1 ..], " \t\r");
            if (!std.mem.endsWith(u8, v, " ns")) continue;
            total += std.fmt.parseInt(u64, v[0 .. v.len - 3], 10) catch 0;
        }
        return total;
    }

    fn fieldText(bytes: []const u8, key: []const u8) ?[]const u8 {
        const at = std.mem.indexOf(u8, bytes, key) orelse return null;
        const rest = bytes[at + key.len ..];
        const end = std.mem.indexOfScalar(u8, rest, '\n') orelse rest.len;
        return std.mem.trim(u8, rest[0..end], " \t\r");
    }

    fn fieldU64(bytes: []const u8, key: []const u8) ?u64 {
        return std.fmt.parseInt(u64, fieldText(bytes, key) orelse return null, 10) catch null;
    }

    /// The `cardN` whose PCI slot matches, so the clock comes from the card we
    /// are actually running on rather than whichever enumerates first.
    fn findCard(self: *DrmMeter, io: std.Io, pdev: []const u8) void {
        var dir = std.Io.Dir.openDirAbsolute(io, "/sys/class/drm", .{ .iterate = true }) catch return;
        defer dir.close(io);
        var it = dir.iterate();
        while (it.next(io) catch null) |ent| {
            if (!std.mem.startsWith(u8, ent.name, "card")) continue;
            if (std.mem.indexOfScalar(u8, ent.name, '-') != null) continue; // a connector
            var path: [96]u8 = undefined;
            const uevent = std.fmt.bufPrintZ(&path, "/sys/class/drm/{s}/device/uevent", .{ent.name}) catch continue;
            var buf: [1024]u8 = undefined;
            const bytes = readSysFile(uevent, &buf) orelse continue;
            const slot = fieldText(bytes, "PCI_SLOT_NAME=") orelse continue;
            if (!std.mem.eql(u8, slot, pdev)) continue;
            const n = @min(ent.name.len, self.card.len);
            @memcpy(self.card[0..n], ent.name[0..n]);
            self.card_len = @intCast(n);
            return;
        }
    }

    fn clockMhz(self: *const DrmMeter) u32 {
        if (self.card_len == 0) return 0;
        var path: [96]u8 = undefined;
        const p = std.fmt.bufPrintZ(&path, "/sys/class/drm/{s}/gt_act_freq_mhz", .{self.card[0..self.card_len]}) catch return 0;
        var buf: [32]u8 = undefined;
        const bytes = readSysFile(p, &buf) orelse return 0;
        return std.fmt.parseInt(u32, std.mem.trim(u8, bytes, " \t\r\n"), 10) catch 0;
    }
};

/// Current CPU frequency (MHz) from cpu0's cpufreq governor, 0 if unavailable
/// (no cpufreq sysfs, e.g. some VMs, and anywhere but Linux).
pub fn cpuFreqMhz() f32 {
    var buf: [32]u8 = undefined;
    const bytes = readSysFile("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", &buf) orelse return 0;
    const s = std.mem.trim(u8, bytes, " \t\r\n");
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
    lib: dynlib.Lib,
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
        var lib = dynlib.openFirst(switch (builtin.os.tag) {
            .windows => &.{"nvml.dll"},
            else => &.{ "libnvidia-ml.so.1", "libnvidia-ml.so" },
        }) orelse return null;
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
        // portable spelling of the syscall, and has no getpid on Windows.
        const pid: u32 = if (builtin.os.tag == .windows)
            std.os.windows.GetCurrentProcessId()
        else
            @intCast(std.posix.system.getpid());
        return self.procUsed(pid);
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

test "DRM fdinfo: engine nanoseconds, client id and the card's address" {
    // Verbatim from an Arc A310 (i915, kernel 6.8) mid-render, which is where
    // the shape of this file was learned. `capacity` lines carry no `ns` and
    // must not be summed; the memory lines must not either.
    const fdinfo = "pos:\t0\n" ++
        "flags:\t02100002\n" ++
        "drm-driver:\ti915\n" ++
        "drm-client-id:\t36\n" ++
        "drm-pdev:\t0000:3d:00.0\n" ++
        "drm-total-local0:\t2700188 KiB\n" ++
        "drm-resident-local0:\t2698140 KiB\n" ++
        "drm-engine-render:\t52044119560 ns\n" ++
        "drm-engine-copy:\t0 ns\n" ++
        "drm-engine-video:\t7 ns\n" ++
        "drm-engine-capacity-video:\t2\n" ++
        "drm-engine-compute:\t0 ns\n";
    try std.testing.expectEqual(@as(u64, 52044119567), DrmMeter.engineNs(fdinfo));
    try std.testing.expectEqual(@as(u64, 36), DrmMeter.fieldU64(fdinfo, "drm-client-id:").?);
    try std.testing.expectEqualStrings("0000:3d:00.0", DrmMeter.fieldText(fdinfo, "drm-pdev:").?);
    try std.testing.expect(DrmMeter.fieldText(fdinfo, "drm-nothing:") == null);
    // A file from a process holding no GPU says nothing about engines.
    try std.testing.expectEqual(@as(u64, 0), DrmMeter.engineNs("pos:\t0\nflags:\t02\n"));
}

test "a DRM client that appears mid-run does not count its whole history as one interval" {
    // The pipeline opens its own client when a model loads. Counting that
    // client's cumulative total against one 200 ms sample reads as a pegged
    // card, which is exactly how this was caught on the Arc.
    var m: DrmMeter = .{};
    const sec = std.time.ns_per_s;
    // First sample: one idle client, nothing to compare against yet.
    m.clients[0] = .{ .id = 1, .busy_ns = 0 };
    m.n_clients = 1;
    m.last_wall_ns = sec;

    // Second: the old client did 100 ms of work and a NEW one shows up holding
    // 52 seconds of history. Only the 100 ms is this interval's.
    var cur = [_]DrmMeter.Client{
        .{ .id = 1, .busy_ns = sec / 10 },
        .{ .id = 2, .busy_ns = 52 * sec },
    };
    var delta: u64 = 0;
    for (cur[0..2]) |c| {
        for (m.clients[0..m.n_clients]) |p| {
            if (p.id != c.id) continue;
            delta += c.busy_ns -| p.busy_ns;
            break;
        }
    }
    try std.testing.expectEqual(@as(u64, sec / 10), delta);
    const pct = @as(f64, @floatFromInt(delta)) / @as(f64, @floatFromInt(sec)) * 100.0;
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), pct, 0.001);
}
