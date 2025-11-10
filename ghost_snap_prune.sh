#!/usr/bin/env bash
set -euo pipefail
keep=${1:-7}
mapfile -t snaps < <(ls -1d ghost_snap_* 2>/dev/null | sort)
cnt=${#snaps[@]}
if (( cnt<=keep )); then echo "Nothing to prune (have $cnt, keep $keep)"; exit 0; fi
del=$((cnt-keep))
for ((i=0;i<del;i++)); do rm -rf "${snaps[$i]}"; done
echo "Pruned $del old snapshots; kept $keep."