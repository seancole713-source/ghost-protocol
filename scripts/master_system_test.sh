#!/bin/bash
# Ghost Master System Test
# Tests all critical subsystems with PASS/FAIL indicators

# Remove set -e to continue on failures
set +e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "╔═══════════════════════════════════════════════╗"
echo "║   GHOST TRADING SYSTEM - MASTER TEST SUITE   ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

BASE_URL="${GHOST_URL:-http://localhost:5000}"
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

# Helper functions
pass() {
  echo -e "${GREEN}✅ PASS${NC} - $1"
  ((PASS_COUNT++))
}

fail() {
  echo -e "${RED}❌ FAIL${NC} - $1"
  ((FAIL_COUNT++))
}

warn() {
  echo -e "${YELLOW}⚠️  WARN${NC} - $1"
  ((WARN_COUNT++))
}

# Test 1: Server Health
echo "=== A. Runtime & Health ==="
if curl -s -f "$BASE_URL/health" | jq -e '.ok == true' > /dev/null 2>&1; then
  pass "Server health check"
else
  fail "Server health check"
fi

# Test 2: Environment variables (check via health - simplified)
if [ -n "$ALPHAVANTAGE_API_KEY" ] && [ -n "$POLYGON_API_KEY" ]; then
  pass "Environment variables loaded"
else
  warn "Some environment variables missing (may be normal)"
fi

# Test 3: AI Memory loaded (check via cockpit)
if curl -s -f "$BASE_URL/api/cockpit" | jq -e '.ai_memory' > /dev/null 2>&1; then
  pass "AI Memory accessible"
else
  warn "AI Memory not in cockpit snapshot (may be normal)"
fi

echo ""
echo "=== B. Data Providers ==="

# Test 4: Price fetching (WOLF)
if curl -s -f "$BASE_URL/api/price/WOLF" | jq -e '.price' > /dev/null 2>&1; then
  pass "Price provider (WOLF)"
else
  fail "Price provider failing"
fi

# Test 5: Secondary symbol check (SPY)
if curl -s -f "$BASE_URL/api/price/SPY" | jq -e '.price' > /dev/null 2>&1; then
  pass "Multi-symbol price fetch (SPY)"
else
  warn "SPY price unavailable (may be after hours)"
fi

# Test 6: Price diagnostics endpoint
if curl -s -f "$BASE_URL/api/price/diagnostics" | jq -e '.cache_size' > /dev/null 2>&1; then
  pass "Price diagnostics endpoint"
else
  warn "Price diagnostics not available"
fi

echo ""
echo "=== C. Persistence ==="

# Test 7: Portfolio state
if curl -s -f "$BASE_URL/api/portfolio" | jq -e '.nav > 0' > /dev/null 2>&1; then
  pass "Portfolio state loaded"
else
  fail "Portfolio state missing"
fi

# Test 8: Position details
if curl -s -f "$BASE_URL/api/portfolio" | jq -e '.positions | length > 0' > /dev/null 2>&1; then
  pass "Portfolio positions present"
else
  warn "No positions (may be normal)"
fi

echo ""
echo "=== D. Forecast System ==="

# Test 9: 48h forecast
if curl -s -f "$BASE_URL/predict/48h" | jq -e '.points | length > 0' > /dev/null 2>&1; then
  pass "48h forecast generation"
else
  fail "Forecast generation"
fi

# Test 10: Forecast points count
POINT_COUNT=$(curl -s -f "$BASE_URL/predict/48h" | jq '.points | length')
if [ "$POINT_COUNT" -ge 20 ]; then
  pass "Forecast data complete ($POINT_COUNT points)"
else
  warn "Forecast has only $POINT_COUNT points (expected 24)"
fi

echo ""
echo "=== E. SSE Streaming ==="

# Test 11: SSE endpoint responds
if timeout 5 curl -s -N "$BASE_URL/api/cockpit/stream" | head -n 1 | grep -q "data:"; then
  pass "SSE stream active"
else
  fail "SSE stream not responding"
fi

# Test 12: SSE contains snapshot_id
if timeout 5 curl -s -N "$BASE_URL/api/cockpit/stream" | head -n 1 | grep -q "snapshot_id"; then
  pass "SSE snapshot structure valid"
else
  fail "SSE snapshot missing snapshot_id"
fi

echo ""
echo "=== F. UI Panels Data ==="

# Test 13: Cockpit API
if curl -s -f "$BASE_URL/api/cockpit" | jq -e '.portfolio' > /dev/null 2>&1; then
  pass "Cockpit API responding"
else
  fail "Cockpit API broken"
fi

# Test 14: News feed
NEWS_COUNT=$(curl -s -f "$BASE_URL/api/cockpit" | jq '.news_relevant | length')
if [ "$NEWS_COUNT" -ge 5 ]; then
  pass "News feed loaded ($NEWS_COUNT items)"
else
  warn "Low news count ($NEWS_COUNT items)"
fi

# Test 15: Market status
if curl -s -f "$BASE_URL/api/cockpit" | jq -e '.market' > /dev/null 2>&1; then
  pass "Market status present"
else
  fail "Market status missing"
fi

# Test 16: Heatmap tiles
TILE_COUNT=$(curl -s -f "$BASE_URL/api/cockpit" | jq '.heatmap.tiles | length')
if [ "$TILE_COUNT" -ge 1 ]; then
  pass "Heatmap tiles ($TILE_COUNT tiles)"
else
  warn "No heatmap tiles"
fi

echo ""
echo "=== G. Telegram (Config Check) ==="

# Test 17: Telegram config (via env check)
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
  pass "Telegram bot token configured"
else
  warn "Telegram bot token missing"
fi

echo ""
echo "=== H. Observability ==="

# Test 18: Prometheus metrics
if curl -s -f "$BASE_URL/metrics" | grep -q "python_gc_objects"; then
  pass "Prometheus metrics exposed"
else
  fail "Prometheus metrics broken"
fi

# Test 19: Diagnostics page (use summary endpoint)
if curl -s -f "$BASE_URL/diagnostics/summary" | jq -e '.uptime_s' > /dev/null 2>&1; then
  pass "Diagnostics endpoint"
else
  warn "Diagnostics summary not available"
fi

echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║            TEST RESULTS SUMMARY               ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}PASS:${NC} $PASS_COUNT"
echo -e "${YELLOW}WARN:${NC} $WARN_COUNT"
echo -e "${RED}FAIL:${NC} $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
  echo "❌ System has failures - review logs"
  exit 1
elif [ $WARN_COUNT -gt 3 ]; then
  echo "⚠️  System has warnings - review configuration"
  exit 2
else
  echo "✅ All critical systems operational"
  exit 0
fi
