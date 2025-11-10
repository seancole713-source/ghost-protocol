#!/usr/bin/env bash
# Minimal WOLF-only quickstart to exercise core endpoints
set -euo pipefail

HOST="${HOST:-http://127.0.0.1:5000}"
QTY="${QTY:-${WOLF_QTY:-}}"
AVG="${AVG:-${WOLF_AVG_COST:-}}"

declare -a AUTH=()
if [[ -n "${GHOST_API_TOKEN:-}" ]]; then
  AUTH=( -H "Authorization: Bearer ${GHOST_API_TOKEN}" )
fi
pp() {
  if command -v jq >/dev/null 2>&1; then
    jq . || true
  else
    cat
  fi
}

say() { echo -e "\n==> $1"; }

say "Health"
curl -sS "$HOST/health" | pp

say "Position (before)"
curl -sS "$HOST/api/position" | pp

if [[ -n "${QTY}" && -n "${AVG}" ]]; then
  say "Setting position (qty=$QTY, avg_cost=$AVG)"
  curl -sS -X POST "$HOST/api/position" \
    -H 'Content-Type: application/json' \
    "${AUTH[@]}" \
    -d "{\"qty\": ${QTY}, \"avg_cost\": ${AVG}}" | pp
else
  echo "(skip) Set QTY and AVG envs to update position, e.g.: QTY=10 AVG=25.50 $0"
fi

say "Alert preview"
curl -sS "$HOST/api/alerts" | pp

say "Dispatch alert (dedupe/throttle)"
curl -sS -X POST "$HOST/api/alerts/dispatch" "${AUTH[@]}" | pp

say "Cockpit snapshot"
if command -v jq >/dev/null 2>&1; then
  curl -sS "$HOST/api/cockpit" | jq '{snapshot_id, ticker, prices, portfolio}'
else
  curl -sS "$HOST/api/cockpit" | head -c 1000; echo
fi

say "Metrics sample"
curl -sS "$HOST/metrics" | grep -E '^(ghost_up|ghost_cockpit_snapshot_failures_total|ghost_cockpit_snapshot_build_seconds_bucket)' | head -n 10 || true

say "Done"
