//! Numeric building blocks shared by all models.

pub const vmath = @import("ops/vmath.zig");
pub const act = @import("ops/act.zig");
pub const norm = @import("ops/norm.zig");
pub const rope = @import("ops/rope.zig");
pub const matmul = @import("ops/matmul.zig");
pub const attention = @import("ops/attention.zig");
pub const convrot = @import("ops/convrot.zig");
pub const w4a8 = @import("ops/w4a8.zig");
pub const nvfp4 = @import("ops/nvfp4.zig");
pub const conv = @import("ops/conv.zig");
pub const cancel = @import("ops/cancel.zig");
pub const pause = @import("ops/pause.zig");

test {
    _ = vmath;
    _ = act;
    _ = norm;
    _ = rope;
    _ = matmul;
    _ = attention;
    _ = convrot;
    _ = w4a8;
    _ = nvfp4;
    _ = conv;
    _ = cancel;
    _ = pause;
}
