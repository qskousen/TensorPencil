//! f32 GEMM kernel module (single entry point; see common.zig).
const common = @import("common.zig");

export fn matmul_f32() callconv(.spirv_kernel) void {
    common.tiledMatmul(false);
}
