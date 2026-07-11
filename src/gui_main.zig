//! tp-gui: desktop conversational image studio (dvui + SDL3).
//! Thin entry point; app logic lives under src/gui/.
const std = @import("std");
const dvui = @import("dvui");
const app = @import("gui/app.zig");

// Route panics and logging through dvui's backend-aware handlers (matches the
// App interface even though we drive the render loop ourselves).
pub const panic = dvui.App.panic;
pub const std_options: std.Options = .{ .logFn = dvui.App.logFn };

pub fn main(init: std.process.Init) !void {
    try app.run(init);
}
