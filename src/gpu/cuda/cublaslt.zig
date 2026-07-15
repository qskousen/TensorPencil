//! cuBLASLt bindings — pure Zig, runtime-loaded via std.DynLib.
//!
//! The 4th backend (`--backend cuda`) reaches ComfyUI-class GEMM speed by
//! calling the same closed math library ComfyUI does. Loaded exactly like the
//! driver (`cu.zig`) and Vulkan (`vk.zig`): `dlopen` the `.so`, hand-declare the
//! C-ABI entry points as `callconv(.c)` externs, no linking / headers / nvcc.
//! This is where the project's pure-Zig line is deliberately crossed (a
//! closed-source math library, not just the driver) — documented in PLAN.md /
//! README; the CPU/Vulkan/zig-cuda backends stay pure and default.
//!
//! cuBLASLt does ONLY the GEMM. The int8 convrot prep (rotate + per-row
//! quantize) and the per-row×per-col rescale stay in our hand-PTX kernels; this
//! module just supplies the matmul in the middle.
//!
//! Opaque handles are `?*anyopaque`; descriptor structs (`MatmulAlgo`,
//! `HeuristicResult`) are `extern struct`s matching cublasLt.h layout. Signatures
//! machine-checked against cublasLt.h (CUDA 13). We prefer the versioned
//! `/usr/local/cuda` `.so.13` ahead of the system `.so.12`.

const std = @import("std");
const cu = @import("cu.zig");

// ---- Handles / opaque descriptor pointers -----------------------------------
pub const Handle = ?*anyopaque;
pub const MatmulDesc = ?*anyopaque;
pub const MatrixLayout = ?*anyopaque;
pub const MatmulPreference = ?*anyopaque;

pub const Status = c_int;
pub const SUCCESS: Status = 0;

/// cublasLtMatmulAlgo_t — 64 opaque bytes the heuristic fills in.
pub const MatmulAlgo = extern struct {
    data: [8]u64 = @splat(0),
};

/// cublasLtMatmulHeuristicResult_t.
pub const HeuristicResult = extern struct {
    algo: MatmulAlgo = .{},
    workspaceSize: usize = 0,
    state: Status = 0,
    wavesCount: f32 = 0,
    reserved: [4]c_int = @splat(0),
};

// ---- Enums we use (values from library_types.h / cublas_api.h / cublasLt.h) --
// cudaDataType_t
pub const R_16F: c_int = 2;
pub const R_16BF: c_int = 14; // CUDA_R_16BF
pub const R_32F: c_int = 0;
pub const R_8I: c_int = 3;
pub const R_32I: c_int = 10;
// cublasComputeType_t
pub const COMPUTE_16F: c_int = 64;
pub const COMPUTE_32F: c_int = 68;
pub const COMPUTE_32I: c_int = 72;
// cublasOperation_t
pub const OP_N: c_int = 0;
pub const OP_T: c_int = 1;
// cublasLtMatmulDescAttributes_t
pub const DESC_TRANSA: c_int = 3;
pub const DESC_TRANSB: c_int = 4;
// cublasLtMatrixLayoutAttribute_t
pub const LAYOUT_ORDER: c_int = 1;
pub const LAYOUT_ROWS: c_int = 2;
pub const LAYOUT_COLS: c_int = 3;
pub const LAYOUT_LD: c_int = 4;
pub const LAYOUT_BATCH_COUNT: c_int = 5;
pub const LAYOUT_STRIDED_BATCH_OFFSET: c_int = 6;
// cublasLtOrder_t
pub const ORDER_COL: c_int = 0;
pub const ORDER_ROW: c_int = 1;
pub const ORDER_COL32: c_int = 2;
pub const ORDER_COL4_4R2_8C: c_int = 3;
pub const ORDER_COL32_2R_4R4: c_int = 4;
// cublasLtMatmulPreferenceAttributes_t (SEARCH_MODE=0, MAX_WORKSPACE_BYTES=1)
pub const PREF_MAX_WORKSPACE_BYTES: c_int = 1;

// ---- Function-pointer types --------------------------------------------------
const PFN_Create = *const fn (*Handle) callconv(.c) Status;
const PFN_Destroy = *const fn (Handle) callconv(.c) Status;
const PFN_GetVersion = *const fn () callconv(.c) usize;
const PFN_GetCudartVersion = *const fn () callconv(.c) usize;
const PFN_MatmulDescCreate = *const fn (*MatmulDesc, c_int, c_int) callconv(.c) Status;
const PFN_MatmulDescDestroy = *const fn (MatmulDesc) callconv(.c) Status;
const PFN_MatmulDescSetAttribute = *const fn (MatmulDesc, c_int, *const anyopaque, usize) callconv(.c) Status;
const PFN_MatrixLayoutCreate = *const fn (*MatrixLayout, c_int, u64, u64, i64) callconv(.c) Status;
const PFN_MatrixLayoutDestroy = *const fn (MatrixLayout) callconv(.c) Status;
const PFN_MatrixLayoutSetAttribute = *const fn (MatrixLayout, c_int, *const anyopaque, usize) callconv(.c) Status;
const PFN_MatmulPreferenceCreate = *const fn (*MatmulPreference) callconv(.c) Status;
const PFN_MatmulPreferenceDestroy = *const fn (MatmulPreference) callconv(.c) Status;
const PFN_MatmulPreferenceSetAttribute = *const fn (MatmulPreference, c_int, *const anyopaque, usize) callconv(.c) Status;
const PFN_MatmulAlgoGetHeuristic = *const fn (
    Handle,
    MatmulDesc,
    MatrixLayout,
    MatrixLayout,
    MatrixLayout,
    MatrixLayout,
    MatmulPreference,
    c_int,
    [*]HeuristicResult,
    *c_int,
) callconv(.c) Status;
const PFN_Matmul = *const fn (
    Handle,
    MatmulDesc,
    *const anyopaque, // alpha
    *const anyopaque, // A
    MatrixLayout,
    *const anyopaque, // B
    MatrixLayout,
    *const anyopaque, // beta
    *const anyopaque, // C
    MatrixLayout,
    *anyopaque, // D
    MatrixLayout,
    ?*const MatmulAlgo,
    ?*anyopaque, // workspace
    usize,
    cu.CUstream,
) callconv(.c) Status;

pub const Error = error{CublasLtError};

/// Resolved cuBLASLt entry points (one per process, owned by the caller).
pub const Api = struct {
    lib: std.DynLib,

    cublasLtCreate: PFN_Create,
    cublasLtDestroy: PFN_Destroy,
    cublasLtGetVersion: PFN_GetVersion,
    cublasLtGetCudartVersion: PFN_GetCudartVersion,
    cublasLtMatmulDescCreate: PFN_MatmulDescCreate,
    cublasLtMatmulDescDestroy: PFN_MatmulDescDestroy,
    cublasLtMatmulDescSetAttribute: PFN_MatmulDescSetAttribute,
    cublasLtMatrixLayoutCreate: PFN_MatrixLayoutCreate,
    cublasLtMatrixLayoutDestroy: PFN_MatrixLayoutDestroy,
    cublasLtMatrixLayoutSetAttribute: PFN_MatrixLayoutSetAttribute,
    cublasLtMatmulPreferenceCreate: PFN_MatmulPreferenceCreate,
    cublasLtMatmulPreferenceDestroy: PFN_MatmulPreferenceDestroy,
    cublasLtMatmulPreferenceSetAttribute: PFN_MatmulPreferenceSetAttribute,
    cublasLtMatmulAlgoGetHeuristic: PFN_MatmulAlgoGetHeuristic,
    cublasLtMatmul: PFN_Matmul,

    /// dlopen cuBLASLt and resolve every symbol. Prefers the versioned
    /// `/usr/local/cuda` `.so.13`, then the plain soname, then `.so.12`.
    pub fn load() Error!Api {
        var lib = std.DynLib.open("/usr/local/cuda/lib64/libcublasLt.so.13") catch
            std.DynLib.open("libcublasLt.so.13") catch
            std.DynLib.open("libcublasLt.so.12") catch
            std.DynLib.open("libcublasLt.so") catch return error.CublasLtError;
        errdefer lib.close();

        var api: Api = undefined;
        api.lib = lib;
        // cuBLASLt symbols are unversioned, so the exported name == the field name.
        inline for (.{
            "cublasLtCreate",
            "cublasLtDestroy",
            "cublasLtGetVersion",
            "cublasLtGetCudartVersion",
            "cublasLtMatmulDescCreate",
            "cublasLtMatmulDescDestroy",
            "cublasLtMatmulDescSetAttribute",
            "cublasLtMatrixLayoutCreate",
            "cublasLtMatrixLayoutDestroy",
            "cublasLtMatrixLayoutSetAttribute",
            "cublasLtMatmulPreferenceCreate",
            "cublasLtMatmulPreferenceDestroy",
            "cublasLtMatmulPreferenceSetAttribute",
            "cublasLtMatmulAlgoGetHeuristic",
            "cublasLtMatmul",
        }) |name| {
            const T = @TypeOf(@field(api, name));
            const sym: [:0]const u8 = name;
            @field(api, name) = api.lib.lookup(T, sym) orelse return error.CublasLtError;
        }
        return api;
    }

    pub fn deinit(self: *Api) void {
        self.lib.close();
    }
};

/// Human-readable cuBLASLt status name (cuBLASLt exposes no GetStatusString).
pub fn statusName(s: Status) []const u8 {
    return switch (s) {
        0 => "SUCCESS",
        1 => "NOT_INITIALIZED",
        3 => "ALLOC_FAILED",
        7 => "INVALID_VALUE",
        8 => "ARCH_MISMATCH",
        11 => "MAPPING_ERROR",
        13 => "EXECUTION_FAILED",
        14 => "INTERNAL_ERROR",
        15 => "NOT_SUPPORTED",
        16 => "LICENSE_ERROR",
        else => "UNKNOWN",
    };
}

test "cublasLt loads (skips without the library / a GPU)" {
    var api = Api.load() catch return error.SkipZigTest;
    defer api.deinit();
    // cublasLtGetVersion is context-free — a nonzero version proves the symbols
    // resolved and the ABI is sane.
    try std.testing.expect(api.cublasLtGetVersion() > 0);
}
