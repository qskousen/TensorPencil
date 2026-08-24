//! A tiny algebraic expression evaluator, for expressing a per-layer weight-noise
//! schedule as a curve instead of a constant (see `cuda/wnoise.zig` for what the
//! noise does).
//!
//! The point is the shape. Noise in the last layers and the LM head lands directly
//! on token selection and comes out as gibberish; the same amount early changes
//! what the model is reasoning ABOUT while the late layers still render it as
//! fluent text. `0.08*(1-t)^2` says that in one line, and a constant cannot say it
//! at all.
//!
//! Variables: `t` normalized depth, 0 at the first decoder layer and exactly 1 at
//! the LM head, so a curve that decays in `t` protects the head by construction;
//! `l` the layer index and `n` the layer count, for anyone who wants absolute
//! positions. Functions: abs sqrt exp log sin cos min max clamp pow, plus `pi`.
//! There is no conditional: `min`/`max`/`clamp` cover gating, e.g.
//! `max(0, 0.05*(1 - t/0.3))` is noise on the first 30% of the stack and nothing
//! after it.
//!
//! No allocator and no compiled form: an expression is re-parsed on every
//! evaluation, which is fine because it is evaluated once per layer when the curve
//! is PUBLISHED (a few hundred times on a UI keystroke), never per kernel launch.
//! Callers must therefore never call this from a hot path.

const std = @import("std");

/// What an expression can read.
pub const Vars = struct {
    /// Normalized depth: 0 at the first decoder layer, 1 at the LM head.
    t: f32,
    /// Layer index, and the layer count. `l == n` is the LM head.
    l: f32 = 0,
    n: f32 = 1,
    /// The amount knob, so a curve can be a SHAPE and the amplitude can stay a
    /// single number someone adjusts without editing an expression. Defaults to 1
    /// so a curve written with literal amplitudes (`0.4*(1-t)^2`) means what it
    /// says. A curve is free to ignore it, or to use it in only part of itself
    /// (`a*(1-t)^2 + 0.005` keeps a floor the knob cannot reach).
    a: f32 = 1,
};

pub const Error = error{BadExpression};

/// Evaluate `expr` at `vars`. Returns `error.BadExpression` for anything that is
/// not a complete, well-formed expression; a well-formed one that overflows or
/// divides by zero yields inf or nan, which callers are expected to reject (see
/// `sanitize`).
pub fn eval(expr: []const u8, vars: Vars) Error!f32 {
    var p: Parser = .{ .src = expr, .vars = vars };
    const v = try p.expr(0);
    p.skipSpace();
    if (p.i != p.src.len) return error.BadExpression; // trailing junk
    return v;
}

/// Whether `expr` parses. Checked at a handful of depths rather than one, so an
/// expression that is only ill-formed for some `t` still fails here.
pub fn validate(expr: []const u8) Error!void {
    for ([_]f32{ 0, 0.25, 0.5, 0.75, 1 }) |t| {
        _ = try eval(expr, .{ .t = t, .l = t * 32, .n = 32, .a = 0.5 });
    }
}

/// Whether the amount knob actually changes this curve. Measured rather than
/// scanned for the token `a`, which would need a tokenizer to tell `a` from `abs`
/// and would still be fooled by a curve where it cancels out. A curve that pins
/// its own amplitude is a legitimate thing to write; the UI dims the knob for it
/// instead of letting it silently do nothing.
pub fn respondsToAmount(expr: []const u8) bool {
    for ([_]f32{ 0, 0.25, 0.5, 0.75, 1 }) |t| {
        const lo = eval(expr, .{ .t = t, .a = 0.1 }) catch return false;
        const hi = eval(expr, .{ .t = t, .a = 0.9 }) catch return false;
        if (lo != hi) return true;
    }
    return false;
}

/// A sigma an evaluation can be trusted with: negatives, nan and inf all become
/// 0 (off) rather than a perturbation nobody asked for, and the ceiling keeps a
/// stray `1/t` from multiplying a weight by 10^38.
pub fn sanitize(v: f32) f32 {
    if (!std.math.isFinite(v) or v <= 0) return 0;
    return @min(v, 1.0);
}

/// Evaluate a curve into `out`, one sigma per depth index. `out.len` is the
/// number of decoder layers PLUS ONE: the last slot is the LM head, at t = 1.
/// A curve that fails to parse leaves every slot 0 (off), which is the right
/// reading of "the user is halfway through typing".
pub fn fill(expr: []const u8, amount: f32, out: []f32) void {
    @memset(out, 0);
    if (out.len < 2) return;
    const n: f32 = @floatFromInt(out.len - 1);
    for (out, 0..) |*o, i| {
        const l: f32 = @floatFromInt(i);
        o.* = sanitize(eval(expr, .{ .t = l / n, .l = l, .n = n, .a = amount }) catch return);
    }
}

const max_depth = 32; // parenthesis nesting, so a hostile string cannot blow the stack

const Parser = struct {
    src: []const u8,
    vars: Vars,
    i: usize = 0,

    fn skipSpace(p: *Parser) void {
        while (p.i < p.src.len and (p.src[p.i] == ' ' or p.src[p.i] == '\t')) p.i += 1;
    }

    fn peek(p: *Parser) ?u8 {
        p.skipSpace();
        return if (p.i < p.src.len) p.src[p.i] else null;
    }

    fn take(p: *Parser, c: u8) bool {
        if (p.peek() == c) {
            p.i += 1;
            return true;
        }
        return false;
    }

    fn expr(p: *Parser, depth: u32) Error!f32 {
        if (depth > max_depth) return error.BadExpression;
        var v = try p.term(depth);
        while (p.peek()) |c| {
            if (c == '+') {
                p.i += 1;
                v += try p.term(depth);
            } else if (c == '-') {
                p.i += 1;
                v -= try p.term(depth);
            } else break;
        }
        return v;
    }

    fn term(p: *Parser, depth: u32) Error!f32 {
        var v = try p.unary(depth);
        while (p.peek()) |c| {
            if (c == '*') {
                p.i += 1;
                v *= try p.unary(depth);
            } else if (c == '/') {
                p.i += 1;
                v /= try p.unary(depth);
            } else break;
        }
        return v;
    }

    fn unary(p: *Parser, depth: u32) Error!f32 {
        if (p.take('-')) return -(try p.unary(depth));
        if (p.take('+')) return p.unary(depth);
        return p.power(depth);
    }

    /// Right-associative, and binding tighter than unary minus on its left, so
    /// `-t^2` is `-(t^2)` and `2^-1` is 0.5.
    fn power(p: *Parser, depth: u32) Error!f32 {
        const base = try p.atom(depth);
        if (!p.take('^')) return base;
        const e = try p.unary(depth);
        return std.math.pow(f32, base, e);
    }

    fn atom(p: *Parser, depth: u32) Error!f32 {
        const c = p.peek() orelse return error.BadExpression;
        if (c == '(') {
            p.i += 1;
            const v = try p.expr(depth + 1);
            if (!p.take(')')) return error.BadExpression;
            return v;
        }
        if (std.ascii.isDigit(c) or c == '.') return p.number();
        if (std.ascii.isAlphabetic(c)) return p.nameOrCall(depth);
        return error.BadExpression;
    }

    fn number(p: *Parser) Error!f32 {
        const start = p.i;
        while (p.i < p.src.len and (std.ascii.isDigit(p.src[p.i]) or p.src[p.i] == '.')) p.i += 1;
        // An exponent, but only when it really is one: `2e-3`. Bare `e` is not a
        // constant here, so `2e` is malformed rather than 2.
        if (p.i < p.src.len and (p.src[p.i] == 'e' or p.src[p.i] == 'E')) {
            var j = p.i + 1;
            if (j < p.src.len and (p.src[j] == '+' or p.src[j] == '-')) j += 1;
            if (j < p.src.len and std.ascii.isDigit(p.src[j])) {
                p.i = j;
                while (p.i < p.src.len and std.ascii.isDigit(p.src[p.i])) p.i += 1;
            }
        }
        return std.fmt.parseFloat(f32, p.src[start..p.i]) catch error.BadExpression;
    }

    fn nameOrCall(p: *Parser, depth: u32) Error!f32 {
        const start = p.i;
        while (p.i < p.src.len and (std.ascii.isAlphanumeric(p.src[p.i]) or p.src[p.i] == '_')) p.i += 1;
        const name = p.src[start..p.i];

        if (!p.take('(')) {
            if (std.mem.eql(u8, name, "t")) return p.vars.t;
            if (std.mem.eql(u8, name, "l")) return p.vars.l;
            if (std.mem.eql(u8, name, "n")) return p.vars.n;
            if (std.mem.eql(u8, name, "a")) return p.vars.a;
            if (std.mem.eql(u8, name, "pi")) return std.math.pi;
            return error.BadExpression;
        }

        var args: [3]f32 = @splat(0);
        var na: usize = 0;
        if (p.peek() != ')') {
            while (true) {
                if (na == args.len) return error.BadExpression;
                args[na] = try p.expr(depth + 1);
                na += 1;
                if (!p.take(',')) break;
            }
        }
        if (!p.take(')')) return error.BadExpression;

        const a = args[0];
        if (na == 1) {
            if (std.mem.eql(u8, name, "abs")) return @abs(a);
            if (std.mem.eql(u8, name, "sqrt")) return @sqrt(a);
            if (std.mem.eql(u8, name, "exp")) return @exp(a);
            if (std.mem.eql(u8, name, "log")) return @log(a);
            if (std.mem.eql(u8, name, "sin")) return @sin(a);
            if (std.mem.eql(u8, name, "cos")) return @cos(a);
        } else if (na == 2) {
            if (std.mem.eql(u8, name, "min")) return @min(a, args[1]);
            if (std.mem.eql(u8, name, "max")) return @max(a, args[1]);
            if (std.mem.eql(u8, name, "pow")) return std.math.pow(f32, a, args[1]);
        } else if (na == 3) {
            if (std.mem.eql(u8, name, "clamp")) return std.math.clamp(a, args[1], args[2]);
        }
        return error.BadExpression;
    }
};

/// The shapes this repo recommends, and the only place they are checked to be
/// well-formed. They also appear in tp-llm's `--weight-noise` help and as tp-gui's
/// shipped curve library (`gui/config.zig` `builtin_curves`, which cannot check
/// them: it imports no evaluator). Keep the three in step.
pub const documented_shapes = [_][]const u8{
    "a", // flat
    "a*(1-t)^2", // front-loaded
    "a*(1-t)^4", // front-loaded, steep
    "max(0, a*(1-t/0.4))", // first 40% of the stack only
    "a*sin(pi*t)^2", // bell
    "a*t^2", // back-loaded; corrupts tokens, kept as the counter-example
};

test "every documented shape parses, and behaves the way it is described" {
    for (documented_shapes) |e| {
        errdefer std.debug.print("documented shape does not parse: {s}\n", .{e});
        try validate(e);
    }
    // The claims the docs make about the two that matter most.
    const front = "0.4*(1-t)^2";
    try std.testing.expect(try eval(front, .{ .t = 0 }) > try eval(front, .{ .t = 1 }));
    try std.testing.expectEqual(@as(f32, 0), try eval(front, .{ .t = 1 })); // head untouched
    const back = "0.2*t^2";
    try std.testing.expect(try eval(back, .{ .t = 1 }) > try eval(back, .{ .t = 0 }));
    // The gate really does stop, and before the halfway point.
    try std.testing.expectEqual(@as(f32, 0), try eval("max(0, 0.6*(1-t/0.4))", .{ .t = 0.45 }));
    // The bell peaks in the middle and vanishes at both ends.
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), try eval("0.3*sin(pi*t)^2", .{ .t = 0.5 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), try eval("0.3*sin(pi*t)^2", .{ .t = 0 }), 1e-6);
}

test "a bare number is a valid curve, so the simple case still reads as one" {
    try std.testing.expectEqual(@as(f32, 0.03), try eval("0.03", .{ .t = 0.5 }));
    try std.testing.expectEqual(@as(f32, 0.03), try eval("  0.03  ", .{ .t = 0.5 }));
    try std.testing.expectEqual(@as(f32, 0.02), try eval("2e-2", .{ .t = 0 }));
}

test "arithmetic, precedence and associativity" {
    try std.testing.expectEqual(@as(f32, 7), try eval("1+2*3", .{ .t = 0 }));
    try std.testing.expectEqual(@as(f32, 9), try eval("(1+2)*3", .{ .t = 0 }));
    try std.testing.expectEqual(@as(f32, 1), try eval("3-1-1", .{ .t = 0 })); // left-assoc
    try std.testing.expectEqual(@as(f32, 512), try eval("2^3^2", .{ .t = 0 })); // right-assoc
    try std.testing.expectEqual(@as(f32, -4), try eval("-2^2", .{ .t = 0 })); // -(2^2)
    try std.testing.expectEqual(@as(f32, 0.5), try eval("2^-1", .{ .t = 0 }));
}

test "the amount knob is a variable a shape may read, or ignore" {
    // The shipped shapes are written in `a`, so the knob scales them.
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), try eval("a*(1-t)^2", .{ .t = 0, .a = 0.4 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), try eval("a*(1-t)^2", .{ .t = 0.5, .a = 0.4 }), 1e-6);
    // Default 1, so a shape written with literal amplitudes means what it says.
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), try eval("0.4*(1-t)^2", .{ .t = 0 }), 1e-6);

    try std.testing.expect(respondsToAmount("a*(1-t)^2"));
    try std.testing.expect(respondsToAmount("max(0, a*(1-t/0.4))"));
    // A pinned amplitude ignores the knob, which the UI must be able to say.
    try std.testing.expect(!respondsToAmount("0.4*(1-t)^2"));
    try std.testing.expect(!respondsToAmount("0.03"));
    // Not fooled by names that merely CONTAIN an a, which is why this measures
    // rather than scanning for the token.
    try std.testing.expect(!respondsToAmount("abs(0-0.2)*max(t,0.1)"));
    // A curve where the knob only moves part of it still counts as responding.
    try std.testing.expect(respondsToAmount("a*(1-t)^2 + 0.005"));
    try std.testing.expectApproxEqAbs(@as(f32, 0.005), try eval("a*(1-t)^2 + 0.005", .{ .t = 1, .a = 0.4 }), 1e-6);
    // Malformed does not respond, rather than erroring at a caller that only
    // wants to know whether to dim a field.
    try std.testing.expect(!respondsToAmount("a*(1-t)^"));
}

test "variables and the front-loaded curve this exists for" {
    // 0.08 at the first layer, 0 at the LM head, quadratic in between.
    const c = "0.08*(1-t)^2";
    try std.testing.expectApproxEqAbs(@as(f32, 0.08), try eval(c, .{ .t = 0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), try eval(c, .{ .t = 0.5 }), 1e-6);
    try std.testing.expectEqual(@as(f32, 0), try eval(c, .{ .t = 1 }));
    // l/n reach the same place from absolute indices.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), try eval("l/n", .{ .t = 0, .l = 30, .n = 60 }), 1e-6);
}

test "functions, including the max() gate that stands in for a conditional" {
    try std.testing.expectEqual(@as(f32, 2), try eval("sqrt(4)", .{ .t = 0 }));
    try std.testing.expectEqual(@as(f32, 3), try eval("max(1,3)", .{ .t = 0 }));
    try std.testing.expectEqual(@as(f32, 1), try eval("min(1,3)", .{ .t = 0 }));
    try std.testing.expectEqual(@as(f32, 2), try eval("clamp(5,0,2)", .{ .t = 0 }));
    try std.testing.expectEqual(@as(f32, 1), try eval("abs(0-1)", .{ .t = 0 }));
    // Noise on the first 30% of the stack and nothing after it.
    const gate = "max(0, 0.05*(1 - t/0.3))";
    try std.testing.expect(try eval(gate, .{ .t = 0.1 }) > 0);
    try std.testing.expectEqual(@as(f32, 0), try eval(gate, .{ .t = 0.5 }));
}

test "malformed expressions are rejected, not silently truncated" {
    const bad = [_][]const u8{
        "", "0.03 0.04", "1+", "*2", "(1", "1)", "sqrt", "sqrt(", "sqrt(1,2)",
        "min(1)", "clamp(1,2)", "x", "2e", "abs()", "1+*2", "--", "()",
    };
    for (bad) |b| {
        errdefer std.debug.print("accepted malformed expression: \"{s}\"\n", .{b});
        try std.testing.expectError(error.BadExpression, eval(b, .{ .t = 0.5 }));
    }
}

test "deep nesting is refused rather than recursing to a stack overflow" {
    var buf: [256]u8 = undefined;
    var n: usize = 0;
    for (0..64) |_| {
        buf[n] = '(';
        n += 1;
    }
    buf[n] = '1';
    n += 1;
    for (0..64) |_| {
        buf[n] = ')';
        n += 1;
    }
    try std.testing.expectError(error.BadExpression, eval(buf[0..n], .{ .t = 0 }));
}

test "sanitize turns every unusable value into off" {
    try std.testing.expectEqual(@as(f32, 0), sanitize(-1));
    try std.testing.expectEqual(@as(f32, 0), sanitize(0));
    try std.testing.expectEqual(@as(f32, 0), sanitize(std.math.nan(f32)));
    try std.testing.expectEqual(@as(f32, 0), sanitize(std.math.inf(f32)));
    try std.testing.expectEqual(@as(f32, 1), sanitize(500)); // clamped, not trusted
    try std.testing.expectEqual(@as(f32, 0.05), sanitize(0.05));
}

test "fill spans first layer to LM head, and the head is exactly t=1" {
    var out: [5]f32 = undefined; // 4 layers + the head
    fill("0.08*(1-t)^2", 1, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.08), out[0], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), out[4]); // the head, protected by construction
    // Monotonically decreasing, which is the whole claim of a front-loaded curve.
    for (out[1..], 0..) |v, i| try std.testing.expect(v <= out[i]);
}

test "fill zeroes everything on a malformed curve, so a half-typed box is off" {
    var out: [5]f32 = @splat(0.5);
    fill("0.08*(1-t)^", 1, &out);
    for (out) |v| try std.testing.expectEqual(@as(f32, 0), v);
}
