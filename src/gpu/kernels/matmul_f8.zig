//! fp8-e4m3 GEMM kernel module (single entry point; see common.zig).
const common = @import("common.zig");

export fn matmul_f8() callconv(.spirv_kernel) void {
    common.tiledMatmul(true);
}
