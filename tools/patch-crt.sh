#!/usr/bin/env bash
# patch-crt.sh — work around a Zig 0.16 self-hosted-linker limitation on hosts
# with a recent glibc/binutils toolchain (e.g. Arch/CachyOS, gcc 16+).
#
# Those toolchains ship C-runtime startup objects (Scrt1.o / crt1.o) that carry
# an SFrame section (.sframe / .rela.sframe) using R_X86_64_PC64 relocations.
# Zig 0.16's self-hosted ELF linker cannot process that relocation type and
# aborts every libc-linked build with:
#
#   fatal linker error: unhandled relocation type R_X86_64_PC64 ... crt1.o:.sframe
#
# SFrame is only stack-unwind metadata for the tiny `_start` stub, so stripping
# it from these objects is harmless. This script builds a private crt directory
# (.zig-crt/) that mirrors the system crt dir but with those two objects
# sframe-stripped, plus a libc description file (.zig-crt/libc.txt). build.zig
# auto-detects .zig-crt/libc.txt and routes all native compiles through it, so a
# plain `zig build` then links cleanly. The directory is host-specific and
# .gitignore'd; re-run this script after a glibc/gcc upgrade to refresh it.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$repo_root/.zig-crt"

# Capture Zig's native libc description; we keep every path it reports and
# override only crt_dir to point at our sframe-stripped mirror below.
native_libc="$(zig libc 2>/dev/null)"
crt_dir="$(cd "$(echo "$native_libc" | sed -n 's/^crt_dir=//p')" && pwd)"
echo "system crt_dir: $crt_dir"

rm -rf "$out"
mkdir -p "$out"

# Mirror every entry of the system crt dir as a symlink: Zig resolves the libc
# shared objects (libc.so, libm.so, ...) from crt_dir too, not just crt objects.
for f in "$crt_dir"/*; do
    ln -s "$f" "$out/$(basename "$f")"
done

# Replace the sframe-carrying startup objects with stripped real copies.
patched=0
for o in Scrt1.o crt1.o gcrt1.o Mcrt1.o crti.o crtn.o; do
    if [ -f "$crt_dir/$o" ]; then
        rm -f "$out/$o"
        objcopy --remove-section .sframe --remove-section .rela.sframe \
            "$crt_dir/$o" "$out/$o"
        patched=$((patched + 1))
    fi
done
echo "stripped .sframe from $patched crt object(s)"

# Reuse Zig's detected paths verbatim, replacing only the crt_dir line.
echo "$native_libc" | sed "s#^crt_dir=.*#crt_dir=$out#" > "$out/libc.txt"

echo "wrote $out/libc.txt"
echo "done — 'zig build' will now pick this up automatically."
