//! f32 GEMM kernel module (single entry point; see common.zig).
const common = @import("common.zig");
const abi = @import("abi.zig");

export fn matmul_f32() callconv(abi.kernel) void {
    common.tiledMatmul(false);
}
