//! The tp-serve protocol layer shared by the daemon and tp-gui: the wire types,
//! certificates, WebSocket framing, the link a byte stream rides, the HTTP
//! client half and the engine-free server half. Depends on tp_core for the
//! sampler vocabulary and on TensorPencil for the pipeline's, and on nothing
//! above it; the tls dependency is only linked where a remote link needs it.
pub const wire = @import("wire.zig");
pub const turn_stats = @import("turn_stats.zig");
pub const x509 = @import("x509.zig");
pub const ws = @import("ws.zig");
pub const queue = @import("queue.zig");
pub const link = @import("link.zig");
pub const httpc = @import("httpc.zig");
pub const server = @import("server.zig");
pub const pairing = @import("pairing.zig");
pub const blob = @import("blob.zig");

test {
    _ = blob;
    _ = @import("privacy_test.zig");
    _ = wire;
    _ = turn_stats;
    _ = x509;
    _ = ws;
    _ = queue;
    _ = link;
    _ = httpc;
    _ = server;
    _ = pairing;
}
