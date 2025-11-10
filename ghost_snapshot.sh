#!/usr/bin/env bash
set -euo pipefail
HOST=${HOST:-http://127.0.0.1:5000}
AUTH="Authorization: Bearer ${GHOST_API_TOKEN:-}"
SNAP="ghost_snap_$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$SNAP"
curl -s -H "$AUTH" "$HOST/source/status" > "$SNAP/source_status.json"
curl -s -H "$AUTH" "$HOST/diagnostics"  > "$SNAP/diagnostics.json"
curl -s -H "$AUTH" "$HOST/portfolio"    > "$SNAP/portfolio.json"
curl -s -H "$AUTH" "$HOST/fusionai"     > "$SNAP/fusionai.json"
curl -s -H "$AUTH" "$HOST/stocks"       > "$SNAP/stocks.json" || true
curl -s -H "$AUTH" "$HOST/news"         > "$SNAP/news.json"   || true
curl -s -H "$AUTH" "$HOST/goals?horizon=daily" > "$SNAP/goals_daily.json"
curl -s -H "$AUTH" "$HOST/risk"         > "$SNAP/risk.json"   || true
curl -s -H "$AUTH" "$HOST/advisor/enhanced" > "$SNAP/advisor.json" || true
curl -s -H "$AUTH" "$HOST/rpc/usage"    > "$SNAP/rpc_usage.json" || true
echo "Snapshot: $SNAP"