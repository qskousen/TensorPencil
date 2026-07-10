//! CUDA Driver API bindings — pure Zig, runtime-loaded via std.DynLib.
//!
//! Mirrors the pure-Zig `vk.zig` Vulkan loader: `libcuda.so.1` is a system
//! driver we `dlopen` (no linking, no CUDA toolkit, no nvcc), and PTX is device
//! IR we hand-emit exactly like SPIR-V (see `coopmat.zig` / `ptx.zig`). This
//! keeps the "pure Zig, no C deps" line by the project's own standard.
//!
//! Signatures/enum values were machine-extracted from the CUDA 13.0 driver
//! header (`cuda.h`). Two ABI hazards baked in here:
//!   * CUdeviceptr is `unsigned long long` (u64) on LP64 — a u32 corrupts every
//!     pointer-carrying call.
//!   * Several entry points export ABI-versioned symbols (`_v2`) that differ
//!     from the documented name (cuMemAlloc -> cuMemAlloc_v2, etc.); the exact
//!     strings are in `load()`.

const std = @import("std");

// ---- Handle / scalar typedefs ------------------------------------------------
pub const CUdeviceptr = u64;
pub const CUdevice = c_int;
pub const CUcontext = ?*anyopaque;
pub const CUmodule = ?*anyopaque;
pub const CUgraph = ?*anyopaque;
pub const CUgraphExec = ?*anyopaque;
pub const CUfunction = ?*anyopaque;
pub const CUstream = ?*anyopaque;
pub const CUevent = ?*anyopaque;
pub const CUresult = c_int;

// ---- CUresult codes ----------------------------------------------------------
pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
pub const CUDA_ERROR_DEINITIALIZED: CUresult = 4;
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
pub const CUDA_ERROR_INVALID_IMAGE: CUresult = 200;
pub const CUDA_ERROR_INVALID_CONTEXT: CUresult = 201;
pub const CUDA_ERROR_NO_BINARY_FOR_GPU: CUresult = 209;
pub const CUDA_ERROR_INVALID_PTX: CUresult = 218;
pub const CUDA_ERROR_INVALID_HANDLE: CUresult = 400;
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;
pub const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: CUresult = 701;
pub const CUDA_ERROR_LAUNCH_TIMEOUT: CUresult = 702;
pub const CUDA_ERROR_LAUNCH_FAILED: CUresult = 719;
pub const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;
pub const CUDA_ERROR_UNKNOWN: CUresult = 999;

// ---- CUjit_option ------------------------------------------------------------
pub const CU_JIT_MAX_REGISTERS: c_int = 0;
pub const CU_JIT_THREADS_PER_BLOCK: c_int = 1;
pub const CU_JIT_WALL_TIME: c_int = 2;
pub const CU_JIT_INFO_LOG_BUFFER: c_int = 3;
pub const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_int = 4;
pub const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
pub const CU_JIT_OPTIMIZATION_LEVEL: c_int = 7;
pub const CU_JIT_TARGET_FROM_CUCONTEXT: c_int = 8;
pub const CU_JIT_TARGET: c_int = 9;
pub const CU_JIT_LOG_VERBOSE: c_int = 12;
pub const CU_JIT_GENERATE_LINE_INFO: c_int = 13;

// ---- CUfunction_attribute ----------------------------------------------------
pub const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 0;
pub const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: c_int = 1;
pub const CU_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
pub const CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: c_int = 9;

// ---- CUdevice_attribute ------------------------------------------------------
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: c_int = 13;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: c_int = 81;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;

// ---- CUctx_flags -------------------------------------------------------------
pub const CU_EVENT_DISABLE_TIMING: c_uint = 0x02;
pub const CU_MEMHOSTREGISTER_PORTABLE: c_uint = 0x01;
pub const CU_MEMHOSTREGISTER_READ_ONLY: c_uint = 0x08;
pub const CU_CTX_SCHED_AUTO: c_uint = 0x00;
pub const CU_CTX_SCHED_SPIN: c_uint = 0x01;
pub const CU_CTX_SCHED_BLOCKING_SYNC: c_uint = 0x04;

// ---- CUfunc_cache ------------------------------------------------------------
pub const CU_FUNC_CACHE_PREFER_NONE: c_int = 0x00;
pub const CU_FUNC_CACHE_PREFER_SHARED: c_int = 0x01;
pub const CU_FUNC_CACHE_PREFER_L1: c_int = 0x02;
pub const CU_FUNC_CACHE_PREFER_EQUAL: c_int = 0x03;

// ---- Function-pointer types --------------------------------------------------
const PFN_cuInit = *const fn (c_uint) callconv(.c) CUresult;
const PFN_cuDriverGetVersion = *const fn (*c_int) callconv(.c) CUresult;
const PFN_cuDeviceGet = *const fn (*CUdevice, c_int) callconv(.c) CUresult;
const PFN_cuDeviceGetCount = *const fn (*c_int) callconv(.c) CUresult;
const PFN_cuDeviceGetName = *const fn ([*]u8, c_int, CUdevice) callconv(.c) CUresult;
const PFN_cuDeviceGetAttribute = *const fn (*c_int, c_int, CUdevice) callconv(.c) CUresult;
const PFN_cuDeviceTotalMem = *const fn (*usize, CUdevice) callconv(.c) CUresult;
const PFN_cuCtxCreate = *const fn (*CUcontext, c_uint, CUdevice) callconv(.c) CUresult;
const PFN_cuCtxDestroy = *const fn (CUcontext) callconv(.c) CUresult;
const PFN_cuCtxSetCurrent = *const fn (CUcontext) callconv(.c) CUresult;
const PFN_cuCtxSynchronize = *const fn () callconv(.c) CUresult;
const PFN_cuCtxSetCacheConfig = *const fn (c_int) callconv(.c) CUresult;
const PFN_cuMemAlloc = *const fn (*CUdeviceptr, usize) callconv(.c) CUresult;
const PFN_cuMemFree = *const fn (CUdeviceptr) callconv(.c) CUresult;
const PFN_cuMemcpyHtoD = *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult;
const PFN_cuMemcpyHtoDAsync = *const fn (CUdeviceptr, *const anyopaque, usize, CUstream) callconv(.c) CUresult;
const PFN_cuMemcpyDtoH = *const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult;
const PFN_cuMemcpyDtoD = *const fn (CUdeviceptr, CUdeviceptr, usize) callconv(.c) CUresult;
const PFN_cuMemHostRegister = *const fn (*anyopaque, usize, c_uint) callconv(.c) CUresult;
const PFN_cuMemHostUnregister = *const fn (*anyopaque) callconv(.c) CUresult;
const PFN_cuMemAllocHost = *const fn (*?*anyopaque, usize) callconv(.c) CUresult;
const PFN_cuMemFreeHost = *const fn (*anyopaque) callconv(.c) CUresult;
const PFN_cuStreamWaitEvent = *const fn (CUstream, CUevent, c_uint) callconv(.c) CUresult;
const PFN_cuEventQuery = *const fn (CUevent) callconv(.c) CUresult;
const PFN_cuMemsetD8 = *const fn (CUdeviceptr, u8, usize) callconv(.c) CUresult;
const PFN_cuMemsetD32 = *const fn (CUdeviceptr, c_uint, usize) callconv(.c) CUresult;
const PFN_cuMemGetInfo = *const fn (*usize, *usize) callconv(.c) CUresult;
const PFN_cuModuleLoadDataEx = *const fn (*CUmodule, *const anyopaque, c_uint, ?[*]c_int, ?[*]?*anyopaque) callconv(.c) CUresult;
const PFN_cuModuleUnload = *const fn (CUmodule) callconv(.c) CUresult;
const PFN_cuModuleGetFunction = *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult;
const PFN_cuModuleGetGlobal = *const fn (*CUdeviceptr, *usize, CUmodule, [*:0]const u8) callconv(.c) CUresult;
const PFN_cuStreamBeginCapture = *const fn (CUstream, c_int) callconv(.c) CUresult;
const PFN_cuStreamEndCapture = *const fn (CUstream, *CUgraph) callconv(.c) CUresult;
const PFN_cuGraphInstantiateWithFlags = *const fn (*CUgraphExec, CUgraph, c_ulonglong) callconv(.c) CUresult;
const PFN_cuGraphLaunch = *const fn (CUgraphExec, CUstream) callconv(.c) CUresult;
const PFN_cuGraphDestroy = *const fn (CUgraph) callconv(.c) CUresult;
const PFN_cuGraphExecDestroy = *const fn (CUgraphExec) callconv(.c) CUresult;
const PFN_cuLaunchKernel = *const fn (
    CUfunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CUstream,
    ?[*]?*anyopaque,
    ?[*]?*anyopaque,
) callconv(.c) CUresult;
const PFN_cuFuncSetAttribute = *const fn (CUfunction, c_int, c_int) callconv(.c) CUresult;
const PFN_cuFuncGetAttribute = *const fn (*c_int, c_int, CUfunction) callconv(.c) CUresult;
const PFN_cuFuncSetCacheConfig = *const fn (CUfunction, c_int) callconv(.c) CUresult;
const PFN_cuStreamCreate = *const fn (*CUstream, c_uint) callconv(.c) CUresult;
const PFN_cuStreamSynchronize = *const fn (CUstream) callconv(.c) CUresult;
const PFN_cuStreamDestroy = *const fn (CUstream) callconv(.c) CUresult;
const PFN_cuEventCreate = *const fn (*CUevent, c_uint) callconv(.c) CUresult;
const PFN_cuEventRecord = *const fn (CUevent, CUstream) callconv(.c) CUresult;
const PFN_cuEventSynchronize = *const fn (CUevent) callconv(.c) CUresult;
const PFN_cuEventElapsedTime = *const fn (*f32, CUevent, CUevent) callconv(.c) CUresult;
const PFN_cuEventDestroy = *const fn (CUevent) callconv(.c) CUresult;
const PFN_cuGetErrorString = *const fn (CUresult, *?[*:0]const u8) callconv(.c) CUresult;
const PFN_cuGetErrorName = *const fn (CUresult, *?[*:0]const u8) callconv(.c) CUresult;

pub const Error = error{CudaError};

/// Resolved CUDA Driver API entry points. One instance is loaded per process
/// (owned by the CUDA Context). All fields are non-null after a successful
/// `load()`.
pub const Api = struct {
    lib: std.DynLib,

    cuInit: PFN_cuInit,
    cuDriverGetVersion: PFN_cuDriverGetVersion,
    cuDeviceGet: PFN_cuDeviceGet,
    cuDeviceGetCount: PFN_cuDeviceGetCount,
    cuDeviceGetName: PFN_cuDeviceGetName,
    cuDeviceGetAttribute: PFN_cuDeviceGetAttribute,
    cuDeviceTotalMem: PFN_cuDeviceTotalMem,
    cuCtxCreate: PFN_cuCtxCreate,
    cuCtxDestroy: PFN_cuCtxDestroy,
    cuCtxSetCurrent: PFN_cuCtxSetCurrent,
    cuCtxSynchronize: PFN_cuCtxSynchronize,
    cuCtxSetCacheConfig: PFN_cuCtxSetCacheConfig,
    cuMemAlloc: PFN_cuMemAlloc,
    cuMemFree: PFN_cuMemFree,
    cuMemcpyHtoD: PFN_cuMemcpyHtoD,
    cuMemcpyHtoDAsync: PFN_cuMemcpyHtoDAsync,
    cuMemcpyDtoH: PFN_cuMemcpyDtoH,
    cuMemcpyDtoD: PFN_cuMemcpyDtoD,
    cuMemHostRegister: PFN_cuMemHostRegister,
    cuMemHostUnregister: PFN_cuMemHostUnregister,
    cuMemAllocHost: PFN_cuMemAllocHost,
    cuMemFreeHost: PFN_cuMemFreeHost,
    cuStreamWaitEvent: PFN_cuStreamWaitEvent,
    cuEventQuery: PFN_cuEventQuery,
    cuMemsetD8: PFN_cuMemsetD8,
    cuMemsetD32: PFN_cuMemsetD32,
    cuMemGetInfo: PFN_cuMemGetInfo,
    cuModuleLoadDataEx: PFN_cuModuleLoadDataEx,
    cuModuleUnload: PFN_cuModuleUnload,
    cuModuleGetFunction: PFN_cuModuleGetFunction,
    cuModuleGetGlobal: PFN_cuModuleGetGlobal,
    cuStreamBeginCapture: PFN_cuStreamBeginCapture,
    cuStreamEndCapture: PFN_cuStreamEndCapture,
    cuGraphInstantiateWithFlags: PFN_cuGraphInstantiateWithFlags,
    cuGraphLaunch: PFN_cuGraphLaunch,
    cuGraphDestroy: PFN_cuGraphDestroy,
    cuGraphExecDestroy: PFN_cuGraphExecDestroy,
    cuLaunchKernel: PFN_cuLaunchKernel,
    cuFuncSetAttribute: PFN_cuFuncSetAttribute,
    cuFuncGetAttribute: PFN_cuFuncGetAttribute,
    cuFuncSetCacheConfig: PFN_cuFuncSetCacheConfig,
    cuStreamCreate: PFN_cuStreamCreate,
    cuStreamSynchronize: PFN_cuStreamSynchronize,
    cuStreamDestroy: PFN_cuStreamDestroy,
    cuEventCreate: PFN_cuEventCreate,
    cuEventRecord: PFN_cuEventRecord,
    cuEventSynchronize: PFN_cuEventSynchronize,
    cuEventElapsedTime: PFN_cuEventElapsedTime,
    cuEventDestroy: PFN_cuEventDestroy,
    cuGetErrorString: PFN_cuGetErrorString,
    cuGetErrorName: PFN_cuGetErrorName,

    /// dlopen the driver and resolve every symbol. Returns error.CudaError if
    /// the library or any required symbol is missing.
    pub fn load() Error!Api {
        var lib = std.DynLib.open("libcuda.so.1") catch std.DynLib.open("libcuda.so") catch return error.CudaError;
        errdefer lib.close();

        var api: Api = undefined;
        api.lib = lib;

        // Resolve each entry; the exported symbol name is the second arg
        // (note the _v2 suffixes on the versioned entry points).
        inline for (.{
            .{ "cuInit", "cuInit" },
            .{ "cuDriverGetVersion", "cuDriverGetVersion" },
            .{ "cuDeviceGet", "cuDeviceGet" },
            .{ "cuDeviceGetCount", "cuDeviceGetCount" },
            .{ "cuDeviceGetName", "cuDeviceGetName" },
            .{ "cuDeviceGetAttribute", "cuDeviceGetAttribute" },
            .{ "cuDeviceTotalMem", "cuDeviceTotalMem_v2" },
            .{ "cuCtxCreate", "cuCtxCreate_v2" },
            .{ "cuCtxDestroy", "cuCtxDestroy_v2" },
            .{ "cuCtxSetCurrent", "cuCtxSetCurrent" },
            .{ "cuCtxSynchronize", "cuCtxSynchronize" },
            .{ "cuCtxSetCacheConfig", "cuCtxSetCacheConfig" },
            .{ "cuMemAlloc", "cuMemAlloc_v2" },
            .{ "cuMemFree", "cuMemFree_v2" },
            .{ "cuMemcpyHtoD", "cuMemcpyHtoD_v2" },
            .{ "cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2" },
            .{ "cuMemcpyDtoH", "cuMemcpyDtoH_v2" },
            .{ "cuMemcpyDtoD", "cuMemcpyDtoD_v2" },
            .{ "cuMemHostRegister", "cuMemHostRegister_v2" },
            .{ "cuMemHostUnregister", "cuMemHostUnregister" },
            .{ "cuMemAllocHost", "cuMemAllocHost_v2" },
            .{ "cuMemFreeHost", "cuMemFreeHost" },
            .{ "cuStreamWaitEvent", "cuStreamWaitEvent" },
            .{ "cuEventQuery", "cuEventQuery" },
            .{ "cuMemsetD8", "cuMemsetD8_v2" },
            .{ "cuMemsetD32", "cuMemsetD32_v2" },
            .{ "cuMemGetInfo", "cuMemGetInfo_v2" },
            .{ "cuModuleLoadDataEx", "cuModuleLoadDataEx" },
            .{ "cuModuleUnload", "cuModuleUnload" },
            .{ "cuModuleGetFunction", "cuModuleGetFunction" },
            .{ "cuModuleGetGlobal", "cuModuleGetGlobal_v2" },
            .{ "cuStreamBeginCapture", "cuStreamBeginCapture_v2" },
            .{ "cuStreamEndCapture", "cuStreamEndCapture" },
            .{ "cuGraphInstantiateWithFlags", "cuGraphInstantiateWithFlags" },
            .{ "cuGraphLaunch", "cuGraphLaunch" },
            .{ "cuGraphDestroy", "cuGraphDestroy" },
            .{ "cuGraphExecDestroy", "cuGraphExecDestroy" },
            .{ "cuLaunchKernel", "cuLaunchKernel" },
            .{ "cuFuncSetAttribute", "cuFuncSetAttribute" },
            .{ "cuFuncGetAttribute", "cuFuncGetAttribute" },
            .{ "cuFuncSetCacheConfig", "cuFuncSetCacheConfig" },
            .{ "cuStreamCreate", "cuStreamCreate" },
            .{ "cuStreamSynchronize", "cuStreamSynchronize" },
            .{ "cuStreamDestroy", "cuStreamDestroy_v2" },
            .{ "cuEventCreate", "cuEventCreate" },
            .{ "cuEventRecord", "cuEventRecord" },
            .{ "cuEventSynchronize", "cuEventSynchronize" },
            .{ "cuEventElapsedTime", "cuEventElapsedTime" },
            .{ "cuEventDestroy", "cuEventDestroy_v2" },
            .{ "cuGetErrorString", "cuGetErrorString" },
            .{ "cuGetErrorName", "cuGetErrorName" },
        }) |pair| {
            const field = pair[0];
            const sym: [:0]const u8 = pair[1];
            const T = @TypeOf(@field(api, field));
            @field(api, field) = api.lib.lookup(T, sym) orelse return error.CudaError;
        }

        return api;
    }

    pub fn deinit(self: *Api) void {
        self.lib.close();
    }

    /// Human-readable name for a CUresult (static string; empty on failure).
    pub fn errName(self: *const Api, r: CUresult) [:0]const u8 {
        var s: ?[*:0]const u8 = null;
        _ = self.cuGetErrorName(r, &s);
        return if (s) |p| std.mem.span(p) else "";
    }

    pub fn errString(self: *const Api, r: CUresult) [:0]const u8 {
        var s: ?[*:0]const u8 = null;
        _ = self.cuGetErrorString(r, &s);
        return if (s) |p| std.mem.span(p) else "";
    }
};

test "cuda driver loads and reports a device" {
    // Only meaningful where a driver + GPU exist; skip cleanly otherwise.
    var api = Api.load() catch return error.SkipZigTest;
    defer api.deinit();
    if (api.cuInit(0) != CUDA_SUCCESS) return error.SkipZigTest;
    var count: c_int = 0;
    try std.testing.expect(api.cuDeviceGetCount(&count) == CUDA_SUCCESS);
    try std.testing.expect(count >= 1);
}
