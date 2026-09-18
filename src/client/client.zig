//! tp-gui's own state that no engine sees: model selection memory,
//! conversation history, the prompt library. Never dvui.
pub const selection = @import("selection.zig");
pub const history = @import("history.zig");
pub const prompt_history = @import("prompt_history.zig");
pub const mirror = @import("mirror.zig");
pub const save_image = @import("save_image.zig");
pub const remote = @import("remote.zig");
pub const hosts = @import("hosts.zig");
pub const models = @import("models.zig");
pub const sync = @import("sync.zig");
pub const sched = @import("sched.zig");

test {
    _ = sync;
    _ = sched;
    _ = selection;
    _ = history;
    _ = prompt_history;
    _ = mirror;
    _ = save_image;
    _ = remote;
    _ = hosts;
    _ = models;
}
