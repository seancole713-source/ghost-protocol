#!/usr/bin/env bash
# Ghost Production Validation Script - Simplified
# Verifies all critical endpoints and data integrity

set -euo pipefail

GHOST_URL="${GHOST_URL:-http://localhost:5000}"

echo "════════════════════════════════════════════════════════════"
echo "   Ghost Production Validation Report"
echo "   $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""

echo "━━━ 1. Health Check ━━━"
curl -fsS http://localhost:5000/health | jq '.'
echo ""

echo "━━━ 2. Positions API ━━━"
curl -fsS http://localhost:5000/api/positions | jq '.positions'
echo ""

echo "━━━ 3. Cockpit Snapshot ━━━"
echo "--- Prices ---"
curl -fsS http://localhost:5000/api/cockpit | jq '{
  provider: .prices.provider,
  price: .prices.price,
  prev_close: .prices.prev_close
}'
echo ""

echo "--- Portfolio ---"
curl -fsS http://localhost:5000/api/cockpit | jq '{
  symbol: .portfolio.symbol,
  qty: .portfolio.qty,
  avg_cost: .portfolio.avg_cost,
  pnl_abs: .portfolio.pnl_abs,
  pnl_pct: .portfolio.pnl_pct
}'
echo ""

echo "--- KPIs ---"
curl -fsS http://localhost:5000/api/cockpit | jq '.kpis'
echo ""

echo "--- Forecast ---"
curl -fsS http://localhost:5000/api/cockpit | jq '{
  enabled: .forecast_summary.enabled,
  horizon_h: .forecast_summary.horizon_h,
  confidence: .forecast_summary.confidence,
  points_count: (.forecast.points | length)
}'
echo ""

echo "--- Metrics ---"
curl -fsS http://localhost:5000/api/cockpit | jq '.metrics'
echo ""

echo "--- Flags ---"
curl -fsS http://localhost:5000/api/cockpit | jq '.flags'
echo ""

echo "━━━ 4. Price Diagnostics ━━━"
curl -fsS http://localhost:5000/diagnostics/summary | jq '.price_diag'
echo ""

echo "━━━ 5. Events Stream (Sample) ━━━"
timeout 3 curl -fsS --no-buffer http://localhost:5000/events 2>&1 | head -n 10 || echo "(timeout after 3s)"
echo ""

echo "════════════════════════════════════════════════════════════"
echo "✓ Validation complete!"
echo "════════════════════════════════════════════════════════════"
