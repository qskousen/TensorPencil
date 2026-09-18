//! What owns a GPU: the chat session, the diffusion engine, the catalog
//! scanner and the system monitor. Linked by tp-serve and, until tp-gui
//! is a pure client, by tp-gui. Never dvui.
pub const chat = @import("chat.zig");
pub const diffuser = @import("diffuser.zig");
pub const driver = @import("driver.zig");
pub const host = @import("host.zig");
pub const scan = @import("scan.zig");
pub const sysmon = @import("sysmon.zig");

test {
    _ = chat;
    _ = diffuser;
    _ = driver;
    _ = host;
    _ = scan;
    _ = sysmon;
}
