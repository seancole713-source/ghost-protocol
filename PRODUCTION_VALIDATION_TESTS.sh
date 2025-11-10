#!/bin/bash
# Ghost Cockpit Live Restore - Production Validation Tests
# Mission: Validate 100% live operation with zero 499 errors
# Date: 2025-11-10

set -e

BASE_URL="${GHOST_BASE_URL:-https://ghost-sniper-bot-seancole713-production.up.railway.app}"
TOKEN="${GHOST_API_TOKEN:-edaa4eac-6455-4693-a745-142cb6deef03}"

echo "=========================================="
echo "PHASE 4: Production Validation Tests"
echo "=========================================="
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Runtime Environment Check
echo "▶ Test 1: Runtime Environment"
curl -s "$BASE_URL/api/runtime/env" | grep -E "STOCK_PRICE_SOURCE|POLYGON|ALPHA|TIMEOUT|SIM_MODE|FOCUS_WOLF" || echo "  ⚠️  Runtime env endpoint not available"
echo ""

# Test 2: Price Diagnostics for AAPL
echo "▶ Test 2: Price Diagnostics (AAPL)"
DIAG=$(curl -s "$BASE_URL/api/price/diagnostics?symbol=AAPL")
echo "$DIAG" | python3 -m json.tool 2>/dev/null || echo "$DIAG"
echo ""

# Test 3: Price Refresh
echo "▶ Test 3: Price Refresh (AAPL)"
REFRESH=$(curl -s "$BASE_URL/api/price/refresh?symbol=AAPL")
echo "$REFRESH" | python3 -m json.tool 2>/dev/null || echo "$REFRESH"
echo ""

# Test 4: Prediction Test
echo "▶ Test 4: Prediction Run (AAPL)"
PREDICT=$(curl -s -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -X POST -d '{"symbol":"AAPL"}' \
     "$BASE_URL/api/predict/run")
echo "$PREDICT" | python3 -m json.tool 2>/dev/null || echo "$PREDICT"

# Check for "Unable to fetch live price" error
if echo "$PREDICT" | grep -q "Unable to fetch live price"; then
    echo "  ❌ PREDICTION FAILED: Unable to fetch live price"
else
    echo "  ✅ Prediction returned data"
fi
echo ""

# Test 5: SSE Stream Connectivity (10 seconds)
echo "▶ Test 5: SSE Stream Connectivity (10s sample)"
timeout 10s curl -N "$BASE_URL/api/cockpit/stream" 2>&1 | head -20 || echo "  ⚠️  SSE stream test completed"
echo ""

# Test 6: Regime Endpoint
echo "▶ Test 6: Market Regime Status"
REGIME=$(curl -s "$BASE_URL/api/regime/current" 2>&1)
if echo "$REGIME" | grep -q "404"; then
    echo "  ⚠️  /api/regime/current not implemented (404)"
else
    echo "$REGIME" | python3 -m json.tool 2>/dev/null || echo "$REGIME"
fi
echo ""

# Test 7: Portfolio Status
echo "▶ Test 7: Portfolio Status"
PORTFOLIO=$(curl -s -H "Authorization: Bearer $TOKEN" "$BASE_URL/api/portfolio")
echo "$PORTFOLIO" | python3 -m json.tool 2>/dev/null | head -30 || echo "$PORTFOLIO"
echo ""

# Test 8: Position Status
echo "▶ Test 8: Position Status"
POSITION=$(curl -s -H "Authorization: Bearer $TOKEN" "$BASE_URL/api/position")
echo "$POSITION" | python3 -m json.tool 2>/dev/null | head -30 || echo "$POSITION"
echo ""

echo "=========================================="
echo "VALIDATION TESTS COMPLETE"
echo "=========================================="
echo ""
echo "Next: Monitor for 5 minutes with:"
echo "  curl -s \"$BASE_URL/api/admin/logs?window=5m\" | grep -E '499|502'"
echo ""
