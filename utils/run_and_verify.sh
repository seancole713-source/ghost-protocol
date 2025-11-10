#!/usr/bin/env bash
set -euo pipefail

# Start server, wait until healthy, run verifier, then stop background server.
# Usage:
#   ./utils/run_and_verify.sh [URL]
# If URL omitted, defaults to http://127.0.0.1:5000 and starts local server.

BASE_URL=${1:-}
STOP_SERVER=0

if [[ -z "$BASE_URL" ]]; then
  BASE_URL="http://127.0.0.1:5000"
  echo "Starting local server at $BASE_URL ..."
  uvicorn wolf_app:app --host 127.0.0.1 --port 5000 &
  SRV_PID=$!
  STOP_SERVER=1
  trap '[[ $STOP_SERVER -eq 1 ]] && kill $SRV_PID 2>/dev/null || true' EXIT
  for i in {1..60}; do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
fi

GHOST_URL="$BASE_URL" python utils/verify_live.py

if [[ ${STOP_SERVER} -eq 1 ]]; then
  echo "Stopping local server ($SRV_PID) ..."
  kill $SRV_PID 2>/dev/null || true
fi
