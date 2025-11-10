#!/usr/bin/env bash
set -euo pipefail

# Simple backup/restore utility for Ghost WOLF state
# Usage:
#   ./utils/backup_restore.sh backup   # writes to backups/YYYYmmdd-HHMMSS/
#   ./utils/backup_restore.sh restore backups/<stamp>/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
BACKUP_DIR="$ROOT_DIR/backups"

mkdir -p "$BACKUP_DIR"
mkdir -p "$DATA_DIR"

cmd="${1:-}"
case "$cmd" in
  backup)
    ts="$(date +%Y%m%d-%H%M%S)"
    out="$BACKUP_DIR/$ts"
    mkdir -p "$out"
    # Known state files
    for f in wolf_state.json wolf.db goals_log.db.backup ghost_state.json snap_portfolio.json; do
      if [[ -f "$ROOT_DIR/$f" ]]; then cp -f "$ROOT_DIR/$f" "$out/"; fi
      if [[ -f "$DATA_DIR/$f" ]]; then cp -f "$DATA_DIR/$f" "$out/"; fi
    done
    echo "Backup created: $out"
    ;;
  restore)
    src="${2:-}"
    if [[ -z "$src" || ! -d "$src" ]]; then
      echo "Provide a backup folder to restore from (e.g., backups/20240101-000000)" >&2
      exit 1
    fi
    for f in wolf_state.json wolf.db goals_log.db.backup ghost_state.json snap_portfolio.json; do
      if [[ -f "$src/$f" ]]; then cp -f "$src/$f" "$ROOT_DIR/$f"; fi
    done
    echo "Restored from: $src"
    ;;
  *)
    echo "Usage: $0 {backup|restore <folder>}"
    exit 1
    ;;
esac
