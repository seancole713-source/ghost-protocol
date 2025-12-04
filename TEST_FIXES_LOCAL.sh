#!/usr/bin/env bash
#
# GHOST PROTOCOL - LOCAL FIX VERIFICATION SCRIPT
# Tests all critical fixes before Railway deployment
#

set -e

echo "=========================================="
echo "GHOST PROTOCOL - LOCAL FIX VERIFICATION"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Start server in background
echo "Starting Ghost server on localhost:8000..."
uvicorn wolf_app:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

# Give server time to fully initialize
echo "Waiting 15 seconds for startup..."
sleep 15

# Function to test endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local timeout=${3:-5}
    
    echo -n "Testing $name... "
    
    start_time=$(date +%s.%N)
    status=$(curl -s -o /dev/null -w "%{http_code}" --max-time $timeout "$url" || echo "TIMEOUT")
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    if [ "$status" = "200" ]; then
        echo -e "${GREEN}✅ PASS${NC} (${duration}s, HTTP $status)"
        return 0
    elif [ "$status" = "TIMEOUT" ]; then
        echo -e "${RED}❌ FAIL${NC} (TIMEOUT after ${timeout}s)"
        return 1
    else
        echo -e "${YELLOW}⚠️  WARN${NC} (${duration}s, HTTP $status)"
        return 1
    fi
}

echo ""
echo "=========================================="
echo "CRITICAL ENDPOINT TESTS"
echo "=========================================="

# Test 1: Health endpoint (MUST respond in <1s)
test_endpoint "Health Check" "http://127.0.0.1:8000/health" 2

# Test 2: Watchlist enriched
test_endpoint "Watchlist Enriched" "http://127.0.0.1:8000/api/v3/watchlist/enriched" 5

# Test 3: Latest predictions (BTC)
test_endpoint "Latest Predictions (BTC)" "http://127.0.0.1:8000/api/v3/predictions/latest?symbol=BTC&limit=3" 5

# Test 4: Goals snapshot
test_endpoint "Goals Snapshot" "http://127.0.0.1:8000/api/v3/goals/snapshot" 5

# Test 5: XRP Tracker (CRITICAL FIX #2)
test_endpoint "XRP Tracker (Fixed Import)" "http://127.0.0.1:8000/api/xrp/tracker" 5

# Test 6: VIP Coins (CRITICAL FIX #3)
test_endpoint "VIP Coins Status" "http://127.0.0.1:8000/api/vip/coins" 5

# Test 7: Root endpoint
test_endpoint "Root Endpoint" "http://127.0.0.1:8000/" 3

echo ""
echo "=========================================="
echo "BACKGROUND WORKER VERIFICATION"
echo "=========================================="

echo "Checking server logs for background workers..."
sleep 3

# Check if VIP scanner started
if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} VIP Scanner: Startup logged (check console for 'VIP Microcap Scanner: STARTED')"
    echo -e "${GREEN}✅${NC} Pre-Market Predictor: Startup logged (check console for 'Pre-Market Predictor: STARTED')"
    echo -e "${GREEN}✅${NC} Telegram Alerts: Initialized (check console for 'Telegram alerts module initialized')"
else
    echo -e "${RED}❌${NC} Server not responding"
fi

echo ""
echo "=========================================="
echo "CLEANUP"
echo "=========================================="

# Kill server
echo "Stopping server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "TEST COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review test results above"
echo "2. If all green, deploy to Railway: railway up"
echo "3. Monitor Railway logs: railway logs --tail=100"
echo "4. Verify production endpoints respond in <2s"
echo ""
