//! `numpy.random.SeedSequence`, the splittable seed deriver.
//!
//! One consumer: `brownian.zig`, the Brownian-tree noise sampler ComfyUI's SDE samplers
//! draw from. torchsde derives each tree node's RNG seed as
//!
//!     np.random.SeedSequence(entropy=seed, spawn_key=(node, depth), pool_size=24)
//!         .generate_state(4)
//!
//! so a bit-exact `dpmpp_2m_sde` needs a bit-exact SeedSequence. It is a small
//! self-contained integer hash with no dependency on numpy's generators, so porting it
//! is cheap; GUESSING at it is not, because a wrong seed produces perfectly plausible
//! noise that simply is not ComfyUI's.
//!
//! The algorithm (numpy's `bit_generator.pyx`, stable since 1.19): assemble the entropy
//! as little-endian u32 words, then, ONLY if a spawn key is present, zero-pad up to
//! `pool_size` and append the spawn key's words (that conditional padding is numpy's
//! gh-16539 fix, and is why an unspawned sequence and a `spawn_key=(0, 0)` one differ);
//! mix it into a `pool_size`-word pool with `hashmix`/`mix`, sharing one running
//! `hash_const` across the whole pass; then run a second, independent hash
//! (`INIT_B`/`MULT_B`) over the pool cyclically in `generate_state`.
//!
//! Every multiply is u32 wraparound and every shift is by 16. Pinned against numpy
//! 2.2.6.

const std = @import("std");

const init_a: u32 = 0x43b0d7e5;
const mult_a: u32 = 0x931e8875;
const init_b: u32 = 0x8b51f9dd;
const mult_b: u32 = 0x58f38ded;
const mix_mult_l: u32 = 0xca01f9dd;
const mix_mult_r: u32 = 0x4973f715;
const xshift: u5 = 16;

/// numpy's default; torchsde's `BrownianTree` asks for 24.
pub const default_pool_size: usize = 4;
/// Largest pool this implementation will build without an allocator. torchsde uses
/// 24; the cap is generous so a caller cannot silently truncate.
pub const max_pool_size: usize = 64;
/// Assembled entropy is at most `max_pool_size` padding words plus the seed's and
/// spawn key's own words.
const max_entropy_words: usize = max_pool_size + 8;

/// The running `hash_const` is shared state across a whole `mixEntropy` pass, which
/// is why it is threaded through rather than being a pure function of `value`.
fn hashmix(value_in: u32, hash_const: *u32) u32 {
    var value = value_in;
    value ^= hash_const.*;
    hash_const.* *%= mult_a;
    value *%= hash_const.*;
    value ^= value >> xshift;
    return value;
}

fn mix(x: u32, y: u32) u32 {
    var result = mix_mult_l *% x -% mix_mult_r *% y;
    result ^= result >> xshift;
    return result;
}

/// Little-endian u32 words of a non-negative integer. Zero is one word, not
/// zero words (`_int_to_uint32_array` appends before its loop), and that matters:
/// the spawn key `(0, 0)` contributes two words and so triggers the pool padding.
fn intWords(n: u64, out: []u32) usize {
    if (n == 0) {
        out[0] = 0;
        return 1;
    }
    var v = n;
    var i: usize = 0;
    while (v > 0) {
        out[i] = @truncate(v);
        v >>= 32;
        i += 1;
    }
    return i;
}

/// A seeded entropy pool. Construct with `init`, read derived seeds with
/// `generateState`.
pub const SeedSequence = struct {
    pool: [max_pool_size]u32,
    pool_size: usize,

    /// `spawn_key` is torchsde's `(node_index, depth)`; pass `&.{}` for the
    /// top-level sequence. Panics if `pool_size` is out of range rather than
    /// silently producing a different stream.
    pub fn init(entropy: u64, spawn_key: []const u64, pool_size: usize) SeedSequence {
        std.debug.assert(pool_size >= default_pool_size and pool_size <= max_pool_size);

        // --- get_assembled_entropy ---
        var words: [max_entropy_words]u32 = undefined;
        var n = intWords(entropy, &words);

        var key_words: [8]u32 = undefined;
        var n_key: usize = 0;
        for (spawn_key) |k| {
            n_key += intWords(k, key_words[n_key..]);
        }

        if (n_key > 0) {
            // Pad the seed's words out to the pool size *before* appending the
            // spawn key, numpy's gh-16539 fix, applied only when a spawn key is
            // present so unspawned streams stay backwards-compatible. Skipping it
            // makes every non-root tree node draw the wrong noise.
            while (n < pool_size) {
                words[n] = 0;
                n += 1;
            }
            @memcpy(words[n..][0..n_key], key_words[0..n_key]);
            n += n_key;
        }

        // --- mix_entropy ---
        var self: SeedSequence = .{ .pool = undefined, .pool_size = pool_size };
        const mixer = self.pool[0..pool_size];
        const entropy_array = words[0..n];
        var hash_const: u32 = init_a;

        for (mixer, 0..) |*m, i| {
            m.* = hashmix(if (i < entropy_array.len) entropy_array[i] else 0, &hash_const);
        }
        // Mix all bits together so late words can affect earlier ones.
        for (0..pool_size) |i_src| {
            for (0..pool_size) |i_dst| {
                if (i_src != i_dst) {
                    mixer[i_dst] = mix(mixer[i_dst], hashmix(mixer[i_src], &hash_const));
                }
            }
        }
        // Any entropy past the pool size is mixed into every pool word. (Only
        // reachable with a spawn key, which pads the seed out to the pool size and
        // then appends; an unspawned u64 seed is at most two words.)
        if (entropy_array.len > pool_size) {
            for (entropy_array[pool_size..]) |src| {
                for (0..pool_size) |i_dst| {
                    mixer[i_dst] = mix(mixer[i_dst], hashmix(src, &hash_const));
                }
            }
        }
        return self;
    }

    /// Fill `out` with derived 32-bit seeds (numpy's `generate_state(len(out))`).
    pub fn generateState(self: *const SeedSequence, out: []u32) void {
        var hash_const: u32 = init_b;
        for (out, 0..) |*o, i| {
            var data_val = self.pool[i % self.pool_size];
            data_val ^= hash_const;
            hash_const *%= mult_b;
            data_val *%= hash_const;
            data_val ^= data_val >> xshift;
            o.* = data_val;
        }
    }
};

// --- tests -----------------------------------------------------------------

test "matches numpy.random.SeedSequence" {
    // Ground truth from the reference venv (numpy 2.2.6):
    //   np.random.SeedSequence(entropy=E, spawn_key=K, pool_size=24)
    // The pool itself is checked too, not just the derived state: a mismatch in
    // `mix_entropy` and one in `generate_state` are different bugs.
    {
        const ss = SeedSequence.init(12345, &.{}, 24);
        const want_pool = [24]u32{
            3484970244, 1259540094, 766396579,  2580326478, 3023244331, 1866214405,
            4246073539, 2583160767, 3365415950, 3340230016, 2134211590, 3381520003,
            2316768650, 619929141,  3819241767, 3218438552, 3141347870, 2899737015,
            2288355322, 4008591130, 535689912,  131056830,  1015478687, 1687390393,
        };
        try std.testing.expectEqualSlices(u32, &want_pool, ss.pool[0..24]);

        var state: [3]u32 = undefined;
        ss.generateState(&state);
        try std.testing.expectEqualSlices(u32, &.{ 2888090517, 4151165838, 1159728038 }, &state);
    }
    // A spawn key of all zeros is NOT the same as no spawn key, it is what
    // triggers the zero-padding to `pool_size`. torchsde's root split uses
    // exactly `(0, 0)`, so this case is load-bearing.
    {
        const ss = SeedSequence.init(12345, &.{ 0, 0 }, 24);
        var state: [4]u32 = undefined;
        ss.generateState(&state);
        try std.testing.expectEqualSlices(u32, &.{ 3924261748, 2174456085, 2149736121, 3248841855 }, &state);
    }
    {
        const ss = SeedSequence.init(12345, &.{ 3, 2 }, 24);
        var state: [4]u32 = undefined;
        ss.generateState(&state);
        try std.testing.expectEqualSlices(u32, &.{ 961355584, 1002510281, 1773615275, 352994120 }, &state);
    }
}
