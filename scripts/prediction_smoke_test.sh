#!/usr/bin/env bash
#
# Ghost Protocol Prediction System Smoke Test
# Tests prediction endpoints for PACS (stock), BTC/XRP (crypto)
# Validates response structure, timing, and data quality
#

set -euo pipefail

# Configuration
BASE_URL="${GHOST_API_URL:-https://ghost-protocol-production.up.railway.app}"
MAX_DURATION_MS=5000  # 5 seconds max per prediction
TEST_OUTPUT="/tmp/ghost_smoke_test_$(date +%s).json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
TESTS_PASSED=0
TESTS_FAILED=0
ERRORS=()

echo "🚀 Ghost Protocol Prediction Smoke Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Base URL: $BASE_URL"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# Helper: Test an endpoint
test_endpoint() {
    local name="$1"
    local url="$2"
    local required_fields="$3"  # comma-separated
    
    echo -n "Testing $name... "
    
    # Measure timing
    START_MS=$(($(date +%s%N)/1000000))
    
    # Make request
    RESPONSE=$(curl -sS --max-time 10 "$url" 2>&1) || {
        echo -e "${RED}✗ FAILED${NC} (curl error)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        ERRORS+=("$name: curl failed")
        return 1
    }
    
    END_MS=$(($(date +%s%N)/1000000))
    DURATION_MS=$((END_MS - START_MS))
    
    # Check if valid JSON
    if ! echo "$RESPONSE" | python3 -m json.tool > /dev/null 2>&1; then
        echo -e "${RED}✗ FAILED${NC} (invalid JSON)"
        echo "Response: $RESPONSE"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        ERRORS+=("$name: invalid JSON")
        return 1
    fi
    
    # Check required fields
    local missing_fields=()
    IFS=',' read -ra FIELDS <<< "$required_fields"
    for field in "${FIELDS[@]}"; do
        if ! echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if '$field' in str(d) else 1)" 2>/dev/null; then
            missing_fields+=("$field")
        fi
    done
    
    if [ ${#missing_fields[@]} -gt 0 ]; then
        echo -e "${RED}✗ FAILED${NC} (missing fields: ${missing_fields[*]})"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        ERRORS+=("$name: missing ${missing_fields[*]}")
        return 1
    fi
    
    # Check timing
    if [ $DURATION_MS -gt $MAX_DURATION_MS ]; then
        echo -e "${YELLOW}⚠ SLOW${NC} (${DURATION_MS}ms > ${MAX_DURATION_MS}ms)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    fi
    
    echo -e "${GREEN}✓ PASSED${NC} (${DURATION_MS}ms)"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    return 0
}

# Test: Health endpoint
test_endpoint "Health Check" \
    "$BASE_URL/health" \
    "status"

# Test: Hunter Feed
test_endpoint "Hunter Feed" \
    "$BASE_URL/api/v3/hunter/feed" \
    "movers,timestamp"

# Test: PACS Prediction (stock)
test_endpoint "PACS Stock Prediction" \
    "$BASE_URL/api/predict/run?symbol=PACS" \
    "ok,direction,confidence"

# Test: BTC Prediction (crypto)
test_endpoint "BTC Crypto Prediction" \
    "$BASE_URL/api/predict/run?symbol=BTC" \
    "ok,direction,confidence"

# Test: XRP Prediction (crypto)
test_endpoint "XRP Crypto Prediction" \
    "$BASE_URL/api/predict/run?symbol=XRP" \
    "ok,direction,confidence"

# Test: Latest predictions for PACS
test_endpoint "Latest PACS Predictions" \
    "$BASE_URL/api/v3/predictions/latest?symbol=PACS&limit=5" \
    "predictions,count"

# Test: Latest predictions for BTC
test_endpoint "Latest BTC Predictions" \
    "$BASE_URL/api/v3/predictions/latest?symbol=BTC&limit=5" \
    "predictions,count"

# Test: Latest predictions for XRP
test_endpoint "Latest XRP Predictions" \
    "$BASE_URL/api/v3/predictions/latest?symbol=XRP&limit=5" \
    "predictions,count"

# Test: Goals snapshot
test_endpoint "Goals Snapshot" \
    "$BASE_URL/api/v3/goals/snapshot" \
    "goals,ghost_score"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test Results Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"

if [ ${#ERRORS[@]} -gt 0 ]; then
    echo ""
    echo "Errors:"
    for error in "${ERRORS[@]}"; do
        echo "  • $error"
    done
fi

echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S %Z')"

# Exit with failure if any tests failed
if [ $TESTS_FAILED -gt 0 ]; then
    exit 1
fi

exit 0
