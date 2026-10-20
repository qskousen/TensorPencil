#!/usr/bin/env bash
# Mirror this working tree, uncommitted changes included, onto a Windows box
# over OpenSSH, for a native build there:
#   tools/sync_windows.sh red@redpc.local [dir]   (dir defaults to tp under the user profile)
#   ssh red@redpc.local 'cd tp; zig build test'
# Sends what git sees (tracked + untracked, not ignored, minus deletions). The
# remote copy is cleared first, keeping only its build cache, packages, output and the
# git-ignored testdata/gpu-tests marker; it refuses while anything runs from that tree.
set -euo pipefail
host=${1:?usage: sync_windows.sh user@host [dir]}
dir=${2:-tp}
cd "$(git rev-parse --show-toplevel)"
tarball=$(mktemp --suffix=.tar)
trap 'rm -f "$tarball"' EXIT
git ls-files -z -co --exclude-standard |
    while IFS= read -r -d '' f; do [ -e "$f" ] && printf '%s\0' "$f"; done |
    tar --null -T - -cf "$tarball"
scp -q "$tarball" "$host:tp-sync.tar"
ssh "$host" "\$ErrorActionPreference='Stop'
\$d = Join-Path \$env:USERPROFILE '$dir'
New-Item -ItemType Directory -Force \$d | Out-Null
\$busy = Get-Process | Where-Object { \$_.Path -and \$_.Path.StartsWith(\$d, 'OrdinalIgnoreCase') }
if (\$busy) { throw ('still running from ' + \$d + ': ' + ((\$busy | ForEach-Object { \$_.Name }) -join ', ')) }
\$marker = Join-Path \$d 'testdata\\gpu-tests'
\$gpu = Test-Path \$marker
Get-ChildItem \$d -Force | Where-Object Name -notin '.zig-cache','zig-pkg','zig-out' | Remove-Item -Recurse -Force
\$t = Join-Path \$env:USERPROFILE 'tp-sync.tar'
tar -xf \$t -C \$d
if (\$LASTEXITCODE -ne 0) { throw 'tar failed' }
if (\$gpu) { New-Item -ItemType File -Force \$marker | Out-Null }
Remove-Item \$t"
echo "synced $(tar -tf "$tarball" | wc -l) files to $host:$dir"
