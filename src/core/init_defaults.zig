//! `var x: T = undefined` skips every `field: T = default` — the defaults only
//! run for struct-literal initialization. A stepper `init` that fills fields one
//! by one therefore leaves any field it forgets holding garbage, which reads as
//! a wrong flag rather than a crash (see ZIG.md).
//!
//! `of(T)` applies the declared defaults and leaves the rest undefined, so an
//! init keeps its field-by-field shape without the class of bug.

const std = @import("std");

pub fn of(comptime T: type) T {
    var v: T = undefined;
    inline for (@typeInfo(T).@"struct".fields) |f| {
        if (f.defaultValue()) |d| @field(v, f.name) = d;
    }
    return v;
}

test "of applies declared defaults and leaves the rest alone" {
    const S = struct {
        set_by_caller: u32,
        flag: bool = false,
        opt: ?u8 = null,
        n: usize = 7,
    };
    var s = of(S);
    s.set_by_caller = 3;
    try std.testing.expectEqual(false, s.flag);
    try std.testing.expectEqual(@as(?u8, null), s.opt);
    try std.testing.expectEqual(@as(usize, 7), s.n);
    try std.testing.expectEqual(@as(u32, 3), s.set_by_caller);
}
