//! GPU compute backend (Vulkan, pure Zig: runtime-loaded loader + Zig-authored
//! SPIR-V kernels).

pub const vk = @import("gpu/vk.zig");
pub const spv = @import("gpu/spv.zig");
pub const hazard = @import("gpu/hazard.zig");
pub const context = @import("gpu/context.zig");
pub const dual_table = @import("gpu/kernels/dual_table.zig");
pub const mem_tag = @import("gpu/mem_tag.zig");
pub const weight_noise = @import("gpu/weight_noise.zig");
pub const pipe_cache = @import("gpu/pipe_cache.zig");
pub const Context = context.Context;

/// Experimental CUDA Driver-API backend (pure Zig: dlopen'd libcuda + hand-
/// emitted PTX). Breaks the Vulkan structural ceilings for the int8 GEMM path.
pub const cuda = @import("gpu/cuda.zig");

// A file's tests run only if something here references it: one reached only through
// a call from another test drops out whenever `-Dtest-filter` skips that caller.
test {
    _ = vk;
    _ = spv;
    _ = hazard;
    _ = context;
    _ = context.coopmat;
    _ = @import("gpu/spirv_asm.zig");
    _ = mem_tag;
    _ = weight_noise;
    _ = pipe_cache;
    _ = cuda;
}
