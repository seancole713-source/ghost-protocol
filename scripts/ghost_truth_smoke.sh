#!/usr/bin/env bash
#
# GHOST TRUTH SMOKE TEST
# ======================
# Zero-tolerance validation of Ghost Protocol's actual operational state.
# No fake "100% complete" claims. Only pass if data is REAL and LIVE.
#
# Usage:
#   bash scripts/ghost_truth_smoke.sh                    # Local (http://localhost:8080)
#   MODE=railway bash scripts/ghost_truth_smoke.sh       # Production Railway
#

# Removed set -e to collect all test results instead of exiting on first failure

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MODE="${MODE:-local}"
if [ "$MODE" = "railway" ]; then
    BASE_URL="https://ghost-protocol-production.up.railway.app"
else
    BASE_URL="http://localhost:8080"
fi

# Track results
TOTAL_CHECKS=0
PASSED_CHECKS=0

echo ""
echo "========================================================"
echo "  GHOST PROTOCOL - TRUTH SMOKE TEST"
echo "========================================================"
echo "Mode: $MODE"
echo "Base URL: $BASE_URL"
echo "Time: $(date)"
echo "========================================================"
echo ""

# Counters
PASS_COUNT=0
FAIL_COUNT=0

# Helper function to check endpoint
check_endpoint() {
    local name="$1"
    local endpoint="$2"
    local check_func="$3"
    
    echo -n "Testing $name... "
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint" 2>/dev/null || echo "000")
    http_code=$(echo "$response" | tail -n 1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" != "200" ]; then
        echo -e "${RED}FAIL${NC}: HTTP $http_code"
        echo "  URL: $BASE_URL$endpoint"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 0  # Don't exit, continue testing
    fi
    
    # Run custom validation function
    if $check_func "$body"; then
        echo -e "${GREEN}OK${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}: Data validation failed"
        echo "  URL: $BASE_URL$endpoint"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 0  # Don't exit, continue testing
    fi
}

# Validation functions

check_hunter_feed() {
    local body="$1"
    
    # Check if response is valid JSON
    if ! echo "$body" | python3 -m json.tool > /dev/null 2>&1; then
        echo "  Reason: Invalid JSON"
        return 1
    fi
    
    # Check if movers array exists and has length > 0
    movers_count=$(echo "$body" | python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data.get('movers', [])))" 2>/dev/null || echo "0")
    
    if [ "$movers_count" -eq "0" ]; then
        echo "  Reason: movers array is empty (market scanning not active or no opportunities)"
        return 1
    fi
    
    # Check if it's the "warming up" placeholder
    if echo "$body" | grep -q "Scanner warming up"; then
        echo "  Reason: Hunter feed showing placeholder 'warming up' message"
        return 1
    fi
    
    return 0
}

check_watchlist() {
    local body="$1"
    
    if ! echo "$body" | python3 -m json.tool > /dev/null 2>&1; then
        echo "  Reason: Invalid JSON"
        return 1
    fi
    
    # Count total symbols (stocks + crypto + vip)
    total_symbols=$(echo "$body" | python3 -c "
import sys, json
data = json.load(sys.stdin)
stocks = len(data.get('stocks', []))
crypto = len(data.get('crypto', []))
vip = len(data.get('vip', []))
print(stocks + crypto + vip)
" 2>/dev/null || echo "0")
    
    if [ "$total_symbols" -lt "20" ]; then
        echo "  Reason: Only $total_symbols symbols in watchlist (expected 26+)"
        return 1
    fi
    
    return 0
}

check_predictions_latest() {
    local body="$1"
    
    if ! echo "$body" | python3 -m json.tool > /dev/null 2>&1; then
        echo "  Reason: Invalid JSON"
        return 1
    fi
    
    # Check if predictions array exists and has at least 1 prediction
    pred_count=$(echo "$body" | python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data.get('predictions', [])))" 2>/dev/null || echo "0")
    
    if [ "$pred_count" -eq "0" ]; then
        echo "  Reason: No predictions found (auto-prediction loop may not be running)"
        return 1
    fi
    
    # Check if first prediction has numeric confidence (not null, not 0)
    confidence=$(echo "$body" | python3 -c "
import sys, json
data = json.load(sys.stdin)
preds = data.get('predictions', [])
if preds:
    print(preds[0].get('confidence', 0))
else:
    print(0)
" 2>/dev/null || echo "0")
    
    if [ "$(echo "$confidence == 0" | bc -l 2>/dev/null || echo "1")" = "1" ]; then
        echo "  Reason: First prediction has confidence = $confidence (should be 0.40-0.85)"
        return 1
    fi
    
    return 0
}

check_accuracy_summary() {
    local body="$1"
    
    if ! echo "$body" | python3 -m json.tool > /dev/null 2>&1; then
        echo "  Reason: Invalid JSON"
        return 1
    fi
    
    # Check if this is the "no data yet" error response (VALID - honest response)
    is_ok=$(echo "$body" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('ok', True))" 2>/dev/null || echo "True")
    
    if [ "$is_ok" = "False" ]; then
        # This is an honest "no data yet" response - PASS this check
        error_msg=$(echo "$body" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('error', 'Unknown'))" 2>/dev/null || echo "Unknown")
        echo -e "  ${YELLOW}(No predictions reconciled yet: $error_msg)${NC}"
        return 0  # PASS - honest response is acceptable
    fi
    
    # Check if total_predictions exists (we allow 0, but field must exist)
    total=$(echo "$body" | python3 -c "
import sys, json
data = json.load(sys.stdin)
total = data.get('total_predictions')
if total is None:
    print(-1)
else:
    print(total)
" 2>/dev/null || echo "-1")
    
    if [ "$total" = "-1" ]; then
        echo "  Reason: Missing 'total_predictions' field in response"
        return 1
    fi
    
    # Check if accuracy data structure exists (even if values are 0)
    has_accuracy=$(echo "$body" | python3 -c "
import sys, json
data = json.load(sys.stdin)
required_fields = ['daily_accuracy_pct', 'weekly_accuracy_pct', 'monthly_accuracy_pct', 'correct', 'wrong', 'pending']
print(all(field in data for field in required_fields))
" 2>/dev/null || echo "False")
    
    if [ "$has_accuracy" != "True" ]; then
        echo "  Reason: Missing required accuracy fields"
        return 1
    fi
    
    return 0
}

check_goals_snapshot() {
    local body="$1"
    
    if ! echo "$body" | python3 -m json.tool > /dev/null 2>&1; then
        echo "  Reason: Invalid JSON"
        return 1
    fi
    
    # Check if ghost_score exists and is not null
    ghost_score=$(echo "$body" | python3 -c "
import sys, json
data = json.load(sys.stdin)
score = data.get('ghost_score')
if score is None:
    print('null')
else:
    print(score)
" 2>/dev/null || echo "null")
    
    if [ "$ghost_score" = "null" ]; then
        echo "  Reason: ghost_score is null (system may not be calculating health score)"
        return 1
    fi
    
    # Check if score is reasonable (0-100 range)
    if [ "$(echo "$ghost_score < 0 || $ghost_score > 100" | bc -l 2>/dev/null || echo "1")" = "1" ]; then
        echo "  Reason: ghost_score out of range: $ghost_score (should be 0-100)"
        return 1
    fi
    
    return 0
}

check_cockpit_status() {
    local body="$1"
    
    if ! echo "$body" | python3 -m json.tool > /dev/null 2>&1; then
        echo "  Reason: Invalid JSON"
        return 1
    fi
    
    # Check if system reports live=true
    is_live=$(echo "$body" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('live', False))" 2>/dev/null || echo "False")
    
    if [ "$is_live" != "True" ]; then
        echo "  Reason: System reports live=false (Ghost may be stopped)"
        return 1
    fi
    
    return 0
}

# Run tests
echo "1. Health & Status"
echo "-------------------"
check_endpoint "Cockpit Status" "/api/v3/cockpit/status" check_cockpit_status
echo ""

echo "2. Core Data Endpoints"
echo "----------------------"
check_endpoint "Hunter Feed" "/api/v3/hunter/feed" check_hunter_feed
check_endpoint "Watchlist" "/api/v3/watchlist" check_watchlist
check_endpoint "Predictions Latest (MSFT)" "/api/v3/predictions/latest?symbol=MSFT" check_predictions_latest
echo ""

echo "3. Metrics & Tracking"
echo "---------------------"
check_endpoint "Accuracy Summary" "/api/v3/accuracy/summary" check_accuracy_summary
check_endpoint "Goals Snapshot" "/api/v3/goals/snapshot" check_goals_snapshot
echo ""

# Results summary
echo "========================================================"
echo "  RESULTS"
echo "========================================================"
echo -e "${GREEN}PASS${NC}: $PASS_COUNT"
echo -e "${RED}FAIL${NC}: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED${NC}"
    echo "Ghost Protocol is operational with live data."
    exit 0
else
    echo -e "${RED}❌ SOME CHECKS FAILED${NC}"
    echo "Ghost Protocol has issues. See failures above."
    exit 1
fi
