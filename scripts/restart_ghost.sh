#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
pkill -f 'uvicorn .*wolf_app:APP' || true
pkill -f 'python3 wolf_app.py' || true
nohup uvicorn wolf_app:APP --host 0.0.0.0 --port "${PORT:-8444}" >/tmp/ghost_uvicorn.out 2>&1 &
echo "Ghost restarted at $(date -Is) on port ${PORT:-8444}"
