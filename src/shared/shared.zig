//! Types both tp-gui and tp-serve hold: the settings, the model catalog
//! and its probe table, tool-call parsing, turn statistics, meter math.
//! Pure std plus TensorPencil and known-folders; never dvui, never an engine.
pub const config = @import("config.zig");
pub const framing = @import("framing.zig");
pub const catalog = @import("catalog.zig");
pub const model_spec = @import("model_spec.zig");
pub const pipeline_map = @import("pipeline_map.zig");
pub const toolcall = @import("toolcall.zig");
pub const turn_stats = @import("serve").turn_stats;
pub const vram_split = @import("vram_split.zig");

test {
    _ = config;
    _ = framing;
    _ = catalog;
    _ = model_spec;
    _ = pipeline_map;
    _ = toolcall;
    _ = turn_stats;
    _ = vram_split;
}
