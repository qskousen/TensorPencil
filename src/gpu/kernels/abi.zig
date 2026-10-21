//! The entry-point calling convention and resource declarations every SPIR-V
//! kernel uses. The compiler emits each buffer's layout (Block, Offset,
//! ArrayStride) and binding from these; nothing is decorated by hand.

const std = @import("std");

/// The workgroup size is a placeholder: `spv.withLocalSize` overwrites it at load
/// with the size the Vulkan side picks for the device.
pub const kernel: std.lang.CallingConvention = .{ .spirv_kernel = .{ .x = 1, .y = 1, .z = 1 } };

/// The storage buffer at `binding` of descriptor set 0. Two declarations may share
/// a binding to view it at two element types.
pub fn buffer(comptime T: type, comptime name: []const u8, comptime binding: u32) *addrspace(.storage_buffer) T {
    return @extern(*addrspace(.storage_buffer) T, .{ .name = name, .decoration = .{ .descriptor = .{ .set = 0, .binding = binding } } });
}

/// The module's push-constant block.
pub fn pushConstants(comptime T: type, comptime name: []const u8) *addrspace(.push_constant) T {
    return @extern(*addrspace(.push_constant) T, .{ .name = name });
}
