#!/usr/bin/env bash
# Zip one release build: the executables, what a Windows build installs beside them
# (the DLLs, vips' loader modules) and the license texts. Debug info stays out.
# usage: tools/package_release.sh <name> <install prefix>   ->  dist/tensorpencil-<name>.zip
set -euo pipefail
name=$1
prefix=$2
stage="dist/stage/tensorpencil-$name"
rm -rf "$stage" "dist/tensorpencil-$name.zip"
mkdir -p "$stage"
cp -r "$prefix/bin/." "$stage/"
rm -f "$stage"/*.pdb
# ELF keeps its DWARF inline; drop it but keep the symbols, so a crash still names functions.
for f in "$stage"/*; do
  if [ -f "$f" ] && head -c 4 "$f" | grep -q $'\x7fELF'; then strip --strip-debug "$f"; fi
done
cp LICENSE README.md "$stage/"
if [ -d "$prefix/share/licenses" ]; then cp -r "$prefix/share/licenses" "$stage/licenses"; fi
(cd dist/stage && zip -qr "../tensorpencil-$name.zip" "tensorpencil-$name")
echo "dist/tensorpencil-$name.zip: $(du -h "dist/tensorpencil-$name.zip" | cut -f1)"
