#!/usr/bin/env bash
# Ghost Smoke Test + Telegram dispatch
# Usage: ./smoke-test.sh [BASE_URL]
# Env: GHOST_API_TOKEN (optional), POLYGON_API_KEY (optional)

set -euo pipefail

BASE_URL=${1:-http://127.0.0.1:5000}
AUTH_HEADER=()
[[ -n "${GHOST_API_TOKEN:-}" ]] && AUTH_HEADER=(-H "Authorization: Bearer ${GHOST_API_TOKEN}")

echo "🚀 Ghost Smoke Test"
echo "→ Base URL: $BASE_URL"
[[ -n "${GHOST_API_TOKEN:-}" ]] && echo "→ Auth: bearer set" || echo "→ Auth: not set (some POSTs may be skipped)"
echo

step() { echo; echo "══ $* ═══════════════════════════════════"; }

step "Health"
curl -sS "$BASE_URL/health" | jq

step "Cockpit snapshot (WOLF-only)"
curl -sS "$BASE_URL/api/cockpit" | jq '{ticker,prices,portfolio:{qty:.portfolio.qty,avg:.portfolio.avg_cost,rows:.portfolio.rows},degraded,flags}'

step "Alert preview"
curl -sS "$BASE_URL/api/alerts" | jq '.signal'

step "News (from cockpit)"
curl -sS "$BASE_URL/api/cockpit" | jq '.news | {note:.note,items:(.items|[.[0:5][]?]|length)}'

step "Metrics (first 20 ghost_* lines)"
curl -sS "$BASE_URL/metrics" | grep -E '^ghost_' | head -20 || true

step "Telegram self-test"
curl -sS "$BASE_URL/alerts/selftest" | jq

step "DISPATCH: send a WOLF signal card to Telegram"
if [[ -n "${GHOST_API_TOKEN:-}" ]]; then
  # Primary path: authenticated dispatch (BUY/SELL/HOLD based on current price/avg)
  set +e
  RESP=$(curl -sS -X POST "${BASE_URL}/api/alerts/dispatch?dry_run=0" "${AUTH_HEADER[@]}" \
         -H 'Content-Type: application/json')
  RC=$?
  set -e
  echo "$RESP" | jq '.'
  if [[ $RC -ne 0 || "$(echo "$RESP" | jq -r '.ok')" != "true" ]]; then
    echo "⚠️ Dispatch failed or throttled. Falling back to STATUS card (/alerts/test)…"
    curl -sS -X POST "$BASE_URL/alerts/test" | jq '.'
  else
    echo "✅ Signal enqueued to Telegram."
  fi
else
  echo "ℹ️ No bearer token; trying STATUS card (/alerts/test)…"
  curl -sS -X POST "$BASE_URL/alerts/test" | jq '.'
fi

step "Optional fallback smoke"
echo "1) First snapshot to seed LKG"; curl -sS "$BASE_URL/api/cockpit" >/dev/null
echo "2) If you temporarily break providers, a follow-up call should respond degraded"
curl -sS "$BASE_URL/api/cockpit" | jq '{snapshot_id,flags,degraded,degraded_reasons}'

echo
echo "✅ Smoke test complete."
