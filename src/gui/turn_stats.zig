//! Per-turn measurement shown in a chat message's footer, and its formatting.
//!
//! Split out of `chat.zig` (which pulls in the whole engine) so the arithmetic
//! and the strings are pure std and unit-testable — the same reason
//! `viewmath.zig` and `toolcall.zig` are separate: the numbers here are easy to
//! get subtly wrong (a rate over the wrong token count, a context readout that
//! shows more tokens than its cap) and impossible to eyeball from a screenshot.
//!
//! Who fills the fields, and when, is `chat.Session`'s business — see
//! `TurnStats` there — but the split matters for reading the footers:
//! prefill/prompt numbers are a property of the USER's turn, generation numbers
//! of the assistant VARIANT that answered it.

const std = @import("std");

pub const TurnStats = struct {
    /// Prompt tokens this USER turn contributed to the context: its text, the
    /// chat template's wrapper around it, and any image pad rows.
    prompt_tokens: usize = 0,
    /// Tokens the model actually PREFILLED for this turn — deliberately not the
    /// same number. A render that strips the previous reply's thought re-prefills
    /// that reply too (so this is larger), and a regenerate that rolls back to a
    /// checkpoint prefills nothing at all (so it is zero).
    prefill_tokens: usize = 0,
    prefill_ns: u64 = 0,
    /// Tokens generated into this (assistant) variant. `decode_tokens` is the
    /// decode-loop forward count the rate is over — one fewer than `gen_tokens`,
    /// since the last prefill chunk's logits produce the first token.
    gen_tokens: usize = 0,
    decode_tokens: usize = 0,
    decode_ns: u64 = 0,
    /// Context length with this message in it, the KV capacity currently
    /// COMMITTED (it grows lazily from 4096 up), and the session's ceiling.
    ctx_tokens: usize = 0,
    ctx_cap: usize = 0,
    ctx_max: usize = 0,

    /// Prompt-processing rate, or null when this turn prefilled nothing.
    pub fn ppRate(self: TurnStats) ?f64 {
        if (self.prefill_tokens == 0 or self.prefill_ns == 0) return null;
        return @as(f64, @floatFromInt(self.prefill_tokens)) / (@as(f64, @floatFromInt(self.prefill_ns)) / 1e9);
    }

    /// Generation rate, or null before the second token (the first one comes out
    /// of the prefill's logits, so one token means zero decode forwards).
    pub fn tgRate(self: TurnStats) ?f64 {
        if (self.decode_tokens == 0 or self.decode_ns == 0) return null;
        return @as(f64, @floatFromInt(self.decode_tokens)) / (@as(f64, @floatFromInt(self.decode_ns)) / 1e9);
    }

    /// Whether there is anything worth drawing under a user message.
    pub fn hasPrompt(self: TurnStats) bool {
        return self.prompt_tokens > 0 or self.prefill_tokens > 0;
    }

    /// Whether there is anything worth drawing under an assistant message.
    pub fn hasGen(self: TurnStats) bool {
        return self.gen_tokens > 0 or self.ctx_max > 0;
    }
};

/// Buffer big enough for either footer at any plausible token count.
pub const buf_len = 192;

/// `12,345` — grouped, because these run to six digits (a 131,072-token cap
/// against a 5,310-token context is unreadable ungrouped).
pub fn grouped(buf: []u8, n: usize) []const u8 {
    var digits: [24]u8 = undefined;
    const s = std.fmt.bufPrint(&digits, "{d}", .{n}) catch return "?";
    var out: usize = 0;
    for (s, 0..) |c, i| {
        const left = s.len - i; // digits still to write, including this one
        if (i > 0 and left % 3 == 0) {
            if (out == buf.len) return buf[0..out];
            buf[out] = ',';
            out += 1;
        }
        if (out == buf.len) return buf[0..out];
        buf[out] = c;
        out += 1;
    }
    return buf[0..out];
}

/// The user message's footer: what this turn cost the prompt, and what the
/// model actually chewed through for it.
///
///     `2,341 tok · 3,102 prefilled @ 1,068 tok/s`
///
/// The two counts differ on purpose (see `TurnStats.prefill_tokens`), so the
/// prefilled figure is only shown when there was one — a regenerate rolls back
/// to a cached boundary and prefills nothing.
pub fn formatUser(buf: []u8, s: TurnStats) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    var nb: [32]u8 = undefined;
    var first = true;
    if (s.prompt_tokens > 0) {
        w.print("{s} tok", .{grouped(&nb, s.prompt_tokens)}) catch {};
        first = false;
    }
    if (s.prefill_tokens > 0) {
        if (!first) w.writeAll(" · ") catch {};
        w.print("{s} prefilled", .{grouped(&nb, s.prefill_tokens)}) catch {};
        if (s.ppRate()) |r| w.print(" @ {s} tok/s", .{grouped(&nb, @intFromFloat(@round(r)))}) catch {};
    }
    return w.buffered();
}

/// The assistant message's footer: the turn's two rates, this reply's size, and
/// where the context now stands against both caps.
///
///     `prefill 1,068 tok/s · 24.3 tok/s · 512 tok · ctx 5,310 / 8,192 of 131,072`
///
/// The two caps are separate numbers on purpose: `ctx_cap` is the KV capacity
/// COMMITTED right now (it starts at 4096 and grows in steps), `ctx_max` the
/// session ceiling. A chat at 3.9k/4k is one token from a growth step, not near
/// its 128k limit.
pub fn formatAssistant(buf: []u8, s: TurnStats) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    var nb: [32]u8 = undefined;
    var first = true;
    if (s.ppRate()) |r| {
        w.print("prefill {s} tok/s", .{grouped(&nb, @intFromFloat(@round(r)))}) catch {};
        first = false;
    }
    if (s.tgRate()) |r| {
        if (!first) w.writeAll(" · ") catch {};
        w.print("{d:.1} tok/s", .{r}) catch {};
        first = false;
    }
    if (s.gen_tokens > 0) {
        if (!first) w.writeAll(" · ") catch {};
        w.print("{s} tok", .{grouped(&nb, s.gen_tokens)}) catch {};
        first = false;
    }
    if (s.ctx_max > 0) {
        if (!first) w.writeAll(" · ") catch {};
        var cb: [32]u8 = undefined;
        var mb: [32]u8 = undefined;
        w.print("ctx {s} / {s} of {s}", .{
            grouped(&nb, s.ctx_tokens),
            grouped(&cb, s.ctx_cap),
            grouped(&mb, s.ctx_max),
        }) catch {};
    }
    return w.buffered();
}

test "grouped inserts separators every three digits" {
    var b: [32]u8 = undefined;
    try std.testing.expectEqualStrings("0", grouped(&b, 0));
    try std.testing.expectEqualStrings("7", grouped(&b, 7));
    try std.testing.expectEqualStrings("999", grouped(&b, 999));
    try std.testing.expectEqualStrings("1,000", grouped(&b, 1000));
    try std.testing.expectEqualStrings("12,345", grouped(&b, 12345));
    try std.testing.expectEqualStrings("131,072", grouped(&b, 131072));
    try std.testing.expectEqualStrings("1,234,567", grouped(&b, 1234567));
}

test "rates are over the right counts" {
    // 512 prompt tokens in 0.5 s, 99 decode forwards in 4 s.
    const s: TurnStats = .{
        .prefill_tokens = 512,
        .prefill_ns = 500 * std.time.ns_per_ms,
        .gen_tokens = 100,
        .decode_tokens = 99,
        .decode_ns = 4 * std.time.ns_per_s,
    };
    try std.testing.expectApproxEqAbs(@as(f64, 1024), s.ppRate().?, 1e-9);
    // The rate is over the FORWARD count, not the token count: 99/4, not 100/4.
    try std.testing.expectApproxEqAbs(@as(f64, 24.75), s.tgRate().?, 1e-9);
}

test "a single generated token has no rate yet" {
    // The first token comes from the prefill's logits — zero decode forwards,
    // so a rate would be a division by an interval nothing happened in.
    const s: TurnStats = .{ .gen_tokens = 1, .decode_tokens = 0, .decode_ns = 3 * std.time.ns_per_ms };
    try std.testing.expect(s.tgRate() == null);
    var b: [buf_len]u8 = undefined;
    try std.testing.expectEqualStrings("1 tok", formatAssistant(&b, s));
}

test "footers omit what was not measured" {
    var b: [buf_len]u8 = undefined;
    // A regenerate: prompt cached, nothing prefilled — no prefill segment, and
    // the user message keeps only its own size.
    const regen: TurnStats = .{
        .gen_tokens = 240,
        .decode_tokens = 239,
        .decode_ns = 10 * std.time.ns_per_s,
        .ctx_tokens = 5310,
        .ctx_cap = 8192,
        .ctx_max = 131072,
    };
    try std.testing.expectEqualStrings(
        "23.9 tok/s · 240 tok · ctx 5,310 / 8,192 of 131,072",
        formatAssistant(&b, regen),
    );
    try std.testing.expectEqualStrings("", formatUser(&b, .{}));
    try std.testing.expectEqualStrings("2,341 tok", formatUser(&b, .{ .prompt_tokens = 2341 }));
}

test "the full footers read as intended" {
    var b: [buf_len]u8 = undefined;
    const s: TurnStats = .{
        .prompt_tokens = 2341,
        .prefill_tokens = 3102,
        .prefill_ns = 2 * std.time.ns_per_s,
        .gen_tokens = 512,
        .decode_tokens = 511,
        .decode_ns = 20 * std.time.ns_per_s,
        .ctx_tokens = 5310,
        .ctx_cap = 8192,
        .ctx_max = 131072,
    };
    try std.testing.expectEqualStrings("2,341 tok · 3,102 prefilled @ 1,551 tok/s", formatUser(&b, s));
    try std.testing.expectEqualStrings(
        "prefill 1,551 tok/s · 25.6 tok/s · 512 tok · ctx 5,310 / 8,192 of 131,072",
        formatAssistant(&b, s),
    );
}

test "a footer never overflows its buffer" {
    var b: [buf_len]u8 = undefined;
    const s: TurnStats = .{
        .prompt_tokens = std.math.maxInt(u32),
        .prefill_tokens = std.math.maxInt(u32),
        .prefill_ns = 1,
        .gen_tokens = std.math.maxInt(u32),
        .decode_tokens = std.math.maxInt(u32),
        .decode_ns = 1,
        .ctx_tokens = std.math.maxInt(u32),
        .ctx_cap = std.math.maxInt(u32),
        .ctx_max = std.math.maxInt(u32),
    };
    try std.testing.expect(formatUser(&b, s).len <= b.len);
    try std.testing.expect(formatAssistant(&b, s).len <= b.len);
}
