//! Test gating for the slow integration suite.
//!
//! `zig build test` runs only the fast CPU unit tests. The slow integration
//! tests — GPU device tests and the real-model LLM/parity tests that load
//! multi-GB checkpoints and run inference in Debug — are gated behind
//! `-Dintegration` (see build.zig). GPU tests self-skip because the device
//! `init` fails in test builds when the flag is off; the heavy real-model tests
//! call `requireIntegration` (or `requireModelFile`) at their top.

const std = @import("std");
const build_options = @import("build_options");

/// Skip the calling test unless the integration suite is enabled.
pub fn requireIntegration() error{SkipZigTest}!void {
    if (!build_options.integration) return error.SkipZigTest;
}

/// Skip unless the integration suite is enabled AND `path` exists (a real-model
/// checkpoint). Combines the `-Dintegration` gate with the pre-existing
/// file-presence check the model tests already did.
pub fn requireModelFile(io: std.Io, path: []const u8) error{SkipZigTest}!void {
    if (!build_options.integration) return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;
}
