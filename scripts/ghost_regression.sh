#!/usr/bin/env bash
set -euo pipefail

# Ghost Protocol Baseline Regression Test
# Validates production deployment against 2025-12-11 baseline (commit 7740c6f6)
# All endpoints must return 200 with <8s response time

# Always run from repo root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=========================================="
echo "🛡️  GHOST BASELINE REGRESSION CHECK"
echo "=========================================="
echo "Baseline: 2025-12-11 (commit 7740c6f6)"
echo "Service: tender-benevolence (Railway)"
echo "=========================================="
echo ""

# Production base URL (Railway)
export BASE_URL="https://ghost-protocol-production.up.railway.app"

# Track failures
FAILURES=0

# Helper function to test endpoint
test_endpoint() {
  local label="$1"
  local endpoint="$2"
  local max_time="${3:-8}"
  
  echo "[TEST] $label"
  echo "  → GET $endpoint"
  
  if response=$(curl --max-time "$max_time" -w "\n__HTTP_CODE__:%{http_code}\n__TIME__:%{time_total}s" -sS "$BASE_URL$endpoint" 2>&1); then
    http_code=$(echo "$response" | grep "__HTTP_CODE__" | cut -d: -f2)
    time_taken=$(echo "$response" | grep "__TIME__" | cut -d: -f2)
    
    if [ "$http_code" = "200" ]; then
      echo "  ✅ PASS (HTTP $http_code, $time_taken)"
    else
      echo "  ❌ FAIL (HTTP $http_code, expected 200)"
      FAILURES=$((FAILURES + 1))
    fi
  else
    echo "  ❌ FAIL (timeout or network error after ${max_time}s)"
    FAILURES=$((FAILURES + 1))
  fi
  echo ""
}

# [1] Core Health & System Status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GROUP 1: Core Health & System Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

test_endpoint "Health Check" "/health" 8
test_endpoint "System Metrics" "/api/v3/health/metrics" 8
test_endpoint "Cockpit Status" "/api/v3/cockpit/status" 8

# [2] Watchlist & Predictions (Core Cockpit APIs)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GROUP 2: Watchlist & Predictions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

test_endpoint "Personal Watchlist" "/api/v3/watchlist/user" 8
test_endpoint "Enriched Watchlist (fallback)" "/api/v3/watchlist/enriched" 8
test_endpoint "Latest Predictions (BTC)" "/api/v3/predictions/latest?symbol=BTC&limit=3" 8

# [3] Trading Goals & Accuracy
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GROUP 3: Trading Goals & Accuracy"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

test_endpoint "Goals Snapshot" "/api/v3/goals/snapshot" 8
test_endpoint "Accuracy Summary" "/api/v3/accuracy/summary" 8

# [4] Live Market Feeds
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GROUP 4: Live Market Feeds"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

test_endpoint "Hunter Feed (Real-time)" "/api/v3/hunter/feed" 8

# [5] Specialized Trackers
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GROUP 5: Specialized Trackers"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

test_endpoint "XRP VIP Tracker" "/api/xrp/tracker" 8
test_endpoint "Presale Watcher" "/api/presale/watch" 8

# [6] Optional Local Dev Check (non-blocking)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  OPTIONAL: Local Dev Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if curl --max-time 3 -sS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
  echo "[INFO] Local dev server is running at localhost:8000"
else
  echo "[INFO] Local dev server not running (ignored for regression)"
fi

echo ""
echo "=========================================="
echo "🛡️  REGRESSION TEST COMPLETE"
echo "=========================================="

if [ $FAILURES -eq 0 ]; then
  echo "✅ ALL TESTS PASSED - Baseline is healthy"
  echo ""
  echo "Production deployment matches 2025-12-11 baseline:"
  echo "  • All core APIs return HTTP 200"
  echo "  • Response times < 8 seconds"
  echo "  • No timeouts or errors"
  echo ""
  exit 0
else
  echo "❌ REGRESSION DETECTED - $FAILURES test(s) failed"
  echo ""
  echo "⚠️  BASELINE COMPROMISED"
  echo ""
  echo "The production deployment does NOT match the 2025-12-11 baseline."
  echo "Review failed endpoints above and restore to working state."
  echo ""
  echo "Expected behavior (from baseline):"
  echo "  • HTTP 200 responses on all core APIs"
  echo "  • Sub-second response times (<1s typical)"
  echo "  • No 499 timeouts or 30s+ delays"
  echo ""
  exit 1
fi
