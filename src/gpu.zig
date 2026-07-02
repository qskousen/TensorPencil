//! GPU compute backend (Vulkan, pure Zig: runtime-loaded loader + Zig-authored
//! SPIR-V kernels).

pub const vk = @import("gpu/vk.zig");
pub const spv = @import("gpu/spv.zig");
pub const context = @import("gpu/context.zig");
pub const Context = context.Context;

test {
    _ = vk;
    _ = spv;
    _ = context;
}
