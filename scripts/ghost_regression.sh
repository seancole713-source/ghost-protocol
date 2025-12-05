#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== GHOST REGRESSION CHECK ==="

# Production base URL (Railway)
export BASE_URL="https://ghost-protocol-production.up.railway.app"

echo ""
echo "[1] Railway healthcheck (should be instant and 200)"
curl --max-time 8 -w "\nHTTP:%{http_code} TIME:%{time_total}s\n" -sS "$BASE_URL/health" || {
  echo "ERROR: /health failed or timed out"
  exit 1
}

echo ""
echo "[2] Core Cockpit APIs (watchlist, predictions, goals)"
curl --max-time 8 -w "\nHTTP:%{http_code} TIME:%{time_total}s\n" -sS "$BASE_URL/api/v3/watchlist/enriched" | head || {
  echo "ERROR: /api/v3/watchlist/enriched failed"
  exit 1
}

curl --max-time 8 -w "\nHTTP:%{http_code} TIME:%{time_total}s\n" -sS "$BASE_URL/api/v3/predictions/latest?symbol=BTC&limit=3" | head || {
  echo "ERROR: /api/v3/predictions/latest for BTC failed"
  exit 1
}

curl --max-time 8 -w "\nHTTP:%{http_code} TIME:%{time_total}s\n" -sS "$BASE_URL/api/v3/goals/snapshot" | head || {
  echo "ERROR: /api/v3/goals/snapshot failed"
  exit 1
}

echo ""
echo "[3] Optional local dev health (does not fail regression if dev is down)"
if curl --max-time 3 -w "\nHTTP:%{http_code} TIME:%{time_total}s\n" -sS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
  echo "Local dev /health OK"
else
  echo "Local dev /health not reachable (ignored for regression)"
fi

echo ""
echo "=== REGRESSION CHECK PASSED ==="
