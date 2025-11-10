#!/usr/bin/env bash
set -euo pipefail

GHOST_URL="${GHOST_URL:-http://127.0.0.1:8444}"
GHOST_TOKEN="${GHOST_API_TOKEN:-edaa4eac-6455-4693-a745-142cb6deef03}"

HDR=(-H "Authorization: Bearer ${GHOST_TOKEN}" -H "Content-Type: application/json")
JQ_OK=1; command -v jq >/dev/null 2>&1 || JQ_OK=0
say(){ printf "\n== %s ==\n" "$*"; }
j(){ if [ $JQ_OK -eq 1 ]; then jq -r; else cat; fi; }
ok=1

say "Context"
echo "GHOST_URL=${GHOST_URL}"
echo "GHOST_TOKEN=${GHOST_TOKEN:0:8}…"

say "Status"
S=$(curl -fsS "${GHOST_URL}/api/status" | j) || ok=0
echo "$S"
TICK=$(printf '%s' "$S" | { jq -r '.tick_count' 2>/dev/null || echo 0; })
AUD=$(printf '%s' "$S" | { jq -r '.audit.ok' 2>/dev/null || echo false; })
[ "$TICK" = "null" ] && TICK=0
[ "$AUD" = "true" ] || ok=0

say "Provider status"
curl -fsS "${GHOST_URL}/api/provider/status" | j || ok=0

say "SSE stream (6s)"
timeout 6 curl -fsNs "${GHOST_URL}/api/cockpit/stream" -H "Authorization: Bearer ${GHOST_TOKEN}" \
  | grep -E '^event:|^data:' | head -20 || { echo "no SSE frames in 6s"; ok=0; }

say "Seed AAPL (manual_seed)"
python3 - <<'PY' || true
try:
    from portfolio_store import PortfolioStore
    s=PortfolioStore(); s.save_price("AAPL",218.42,217.80,"manual_seed")
    print("Seeded AAPL")
except Exception as e:
    print("Seed skipped:", e)
PY

say "Predict AAPL"
curl -fsS "${HDR[@]}" -X POST -d '{"symbol":"AAPL"}' "${GHOST_URL}/api/predict/run" | j || ok=0
say "Latest AAPL"
curl -fsS "${HDR[@]}" "${GHOST_URL}/api/predict/latest?symbol=AAPL" | j || ok=0

say "Predict BTC (crypto)"
if ! curl -fsS "${HDR[@]}" -X POST "${GHOST_URL}/api/crypto/predict/run?symbol=BTC" | j; then
  curl -fsS "${HDR[@]}" -X POST -d '{"symbol":"BTC"}' "${GHOST_URL}/api/predict/run" | j || ok=0
fi
say "Latest BTC"
curl -fsS "${HDR[@]}" "${GHOST_URL}/api/predict/latest?symbol=BTC" | j || ok=0

say "SUMMARY"
if [ $ok -eq 1 ]; then
  printf '%s\n' '{"final_state":"PASS"}' | j; exit 0
else
  printf '%s\n' '{"final_state":"FAIL"}' | j; exit 1
fi
