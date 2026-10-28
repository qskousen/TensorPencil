//! A checkpoint file in either format, opened by content: a leading `GGUF` magic
//! picks the GGUF reader, anything else the safetensors one. Trusting the extension
//! instead turns a misnamed file into a baffling error from the other parser (a
//! GGUF handed to the safetensors reader reports `InvalidHeader`).

const std = @import("std");
const safetensors = @import("safetensors.zig");
const gguf_mod = @import("gguf.zig");
const weights_mod = @import("weights.zig");

pub const Container = union(enum) {
    safetensors: safetensors.SafeTensors,
    gguf: gguf_mod.Gguf,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Container {
        return openIn(gpa, io, std.Io.Dir.cwd(), path);
    }

    pub fn openIn(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) !Container {
        if (try isGgufIn(io, dir, path)) return .{ .gguf = try gguf_mod.Gguf.openIn(gpa, io, dir, path) };
        return .{ .safetensors = try safetensors.SafeTensors.openIn(gpa, io, dir, path) };
    }

    /// `open`, naming the file if it fails. A run opens several checkpoints, so a
    /// bare `error.FileNotFound` does not say which one the user got wrong. Used
    /// wherever the path came from the user; a speculative open of a defaulted
    /// path stays on `open`, where a miss is expected and not reported.
    pub fn openNamed(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Container {
        return open(gpa, io, path) catch |err| {
            std.log.err("cannot open {s}: {t}", .{ path, err });
            return err;
        };
    }

    /// `open`, but reading only the header: what a folder scan uses (see
    /// `SafeTensors.openHeader`). Tensors have shapes and dtypes and no bytes.
    pub fn openHeader(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Container {
        return openHeaderIn(gpa, io, std.Io.Dir.cwd(), path);
    }

    pub fn openHeaderIn(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) !Container {
        if (try isGgufIn(io, dir, path)) return .{ .gguf = try gguf_mod.Gguf.openHeaderIn(gpa, io, dir, path) };
        return .{ .safetensors = try safetensors.SafeTensors.openHeaderIn(gpa, io, dir, path) };
    }

    fn isGgufIn(io: std.Io, dir: std.Io.Dir, path: []const u8) !bool {
        var magic: [4]u8 = undefined;
        const f = try dir.openFile(io, path, .{ .mode = .read_only });
        defer f.close(io);
        const n = try f.readPositionalAll(io, &magic, 0);
        if (n < magic.len) return error.CheckpointTooSmall;
        return std.mem.eql(u8, &magic, "GGUF");
    }

    pub fn deinit(self: *Container) void {
        switch (self.*) {
            .safetensors => |*st| st.deinit(),
            .gguf => |*g| g.deinit(),
        }
    }

    /// A store view. Takes a pointer because the containers hold interior
    /// pointers into their own mappings, so a `Container` must stay at a stable
    /// address while anything holds a view.
    pub fn store(self: *const Container) weights_mod.WeightStore {
        return switch (self.*) {
            .safetensors => |*st| .{ .safetensors = st },
            .gguf => |*g| .{ .gguf = g },
        };
    }

    /// Tensor-data bytes, what the VRAM pinning policy sizes itself against.
    /// Declared, not mapped, so it is right on a header-only open too.
    pub fn payloadLen(self: *const Container) usize {
        return switch (self.*) {
            .safetensors => |*st| st.payload_len,
            .gguf => |*g| g.payload_len,
        };
    }

    /// Copy the bytes of `src`, a slice of this file's mapping, into `dst` with
    /// positional reads instead of faulting the mapping (see `SafeTensors.readTo`).
    /// False when the bytes cannot be served that way; the caller copies from
    /// the mapping instead.
    pub fn readTo(self: *const Container, dst: []u8, src: []const u8) bool {
        return switch (self.*) {
            .safetensors => |*st| st.readTo(dst, src),
            .gguf => |*g| g.readTo(dst, src),
        };
    }
};

test "open picks the reader by magic, whatever the extension says" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // A minimal safetensors file named like a GGUF.
    const header = "{\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var buf: [8 + header.len + 4]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8..][0..header.len], header);
    @memset(buf[8 + header.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.gguf", .data = &buf });

    var c = try Container.openIn(std.testing.allocator, io, tmp.dir, "model.gguf");
    defer c.deinit();
    try std.testing.expect(c == .safetensors);
    try std.testing.expect(c.store().get("t") != null);
}
