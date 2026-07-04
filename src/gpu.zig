//! GPU compute backend (Vulkan, pure Zig: runtime-loaded loader + Zig-authored
//! SPIR-V kernels).

pub const vk = @import("gpu/vk.zig");
pub const spv = @import("gpu/spv.zig");
pub const context = @import("gpu/context.zig");
pub const Context = context.Context;

/// Experimental CUDA Driver-API backend (pure Zig: dlopen'd libcuda + hand-
/// emitted PTX). Breaks the Vulkan structural ceilings for the int8 GEMM path.
pub const cuda = @import("gpu/cuda.zig");

test {
    _ = vk;
    _ = spv;
    _ = context;
    _ = cuda;
}
