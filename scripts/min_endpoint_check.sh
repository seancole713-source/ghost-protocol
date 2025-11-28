#!/usr/bin/env bash
set -euo pipefail

BASE="https://ghost-protocol-production.up.railway.app"

check() {
  local label="$1"
  local path="$2"
  echo "=== $label ($path) ==="
  # 8-second hard timeout, no endless hanging
  if ! out=$(curl -m 8 -sS "$BASE$path" 2>&1); then
    echo "ERROR: curl failed for $path"
    echo "Error output: $out"
    return
  fi
  # Try JSON parse; if it fails, show first chars so we know if it's HTML or something else
  python3 - <<EOF || echo "WARN: non-JSON response for $path: ${out:0:120}"
import json
try:
    data = json.loads("""$out""")
    print("OK JSON keys:", list(data.keys())[:10])
except Exception as e:
    print("JSON_ERROR:", e)
    raise
EOF
  echo
}

check "Health"        "/health"
check "PACS predict"  "/api/predict/run?symbol=PACS"
check "BTC predict"   "/api/predict/run?symbol=BTC"
