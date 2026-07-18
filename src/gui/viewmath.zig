//! Pure zoom/pan math for the image viewer, kept free of dvui types so it is
//! unit-testable (`zig build gui-test`).
const std = @import("std");

/// The viewer draws image-space point `u` (pixels, relative to the image
/// center) at screen position `center + (pan + u) * zoom * scale`, i.e. `pan`
/// is in image pixels and its screen offset scales with the zoom. Given the
/// image point `cur` currently under the cursor, return the pan that keeps it
/// stationary on screen when the zoom changes from `zoom` to `new_zoom`.
///
/// Derivation: (pan' + cur) * new_zoom = (pan + cur) * zoom.
pub fn zoomAt(pan: f32, cur: f32, zoom: f32, new_zoom: f32) f32 {
    return (pan + cur) * (zoom / new_zoom) - cur;
}

/// Screen position of image point `u` under the viewer's render mapping —
/// test-only mirror of `Viewer.renderImageArea`.
fn screenPos(center: f32, pan: f32, u: f32, zoom: f32, scale: f32) f32 {
    return center + (pan + u) * zoom * scale;
}

test "zoomAt keeps the point under the cursor fixed on screen" {
    const cases = [_][5]f32{
        // center, pan, cur, zoom, new_zoom
        .{ 500, 0, 300, 0.4, 0.448 }, // first zoom-in from fit, cursor off-center
        .{ 500, 0, -220, 0.4, 0.357 }, // zoom-out, cursor up-left of center
        .{ 640, 85, 410, 1.0, 1.12 }, // already panned, zoom in
        .{ 640, -130, -50, 2.5, 2.232 }, // zoomed in, zoom back out
        .{ 320, 40, 0, 0.7, 0.784 }, // cursor exactly on the image center
    };
    for (cases) |c| {
        const center, const pan, const cur, const zoom, const nz = c;
        const before = screenPos(center, pan, cur, zoom, 1.5);
        const after = screenPos(center, zoomAt(pan, cur, zoom, nz), cur, nz, 1.5);
        try std.testing.expectApproxEqAbs(before, after, 0.001);
    }
}

test "zoomAt is a no-op when the zoom is unchanged (clamped at min/max)" {
    try std.testing.expectApproxEqAbs(@as(f32, 77.5), zoomAt(77.5, -412, 32.0, 32.0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -3.25), zoomAt(-3.25, 190, 0.05, 0.05), 0.0001);
}

test "zoomAt at the image center with no pan stays centered" {
    try std.testing.expectApproxEqAbs(@as(f32, 0), zoomAt(0, 0, 0.5, 2.0), 0.0001);
}
