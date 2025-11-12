#!/bin/bash
#
# Quick smoke test script for Ghost Cockpit deployment verification
# Run after Railway deployment completes with "Healthcheck succeeded"
#

set -e

# Configuration
export GHOST_BASE_URL="${GHOST_BASE_URL:-https://ghost-sniper-bot-seancole713-production.up.railway.app}"
export GHOST_API_TOKEN="${GHOST_API_TOKEN:-edaa4eac-6455-4693-a745-142cb6deef03}"

echo "========================================"
echo "Ghost Cockpit Deployment Smoke Tests"
echo "========================================"
echo "Base URL: $GHOST_BASE_URL"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0

# Test helper
test_endpoint() {
    local name="$1"
    local url="$2"
    local method="${3:-GET}"
    local data="${4:-}"
    local headers="${5:-}"
    
    echo -n "Testing $name... "
    
    local cmd="curl -s -w '\n%{http_code}' -X $method"
    if [ -n "$headers" ]; then
        cmd="$cmd $headers"
    fi
    if [ -n "$data" ]; then
        cmd="$cmd -d '$data'"
    fi
    cmd="$cmd '$url'"
    
    local response=$(eval $cmd 2>&1)
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n-1)
    
    if [[ "$http_code" =~ ^2 ]]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP $http_code)"
        ((passed++))
        if command -v python3 &> /dev/null; then
            echo "$body" | python3 -m json.tool 2>/dev/null | head -5
        fi
    elif [[ "$http_code" == "404" ]]; then
        echo -e "${RED}✗ FAIL${NC} (HTTP 404 - Route not found)"
        ((failed++))
    elif [[ "$http_code" == "499" ]]; then
        echo -e "${RED}✗ FAIL${NC} (HTTP 499 - Client closed request)"
        ((failed++))
    else
        echo -e "${YELLOW}⚠ WARNING${NC} (HTTP $http_code)"
        echo "$body" | head -3
    fi
    echo ""
}

# Test 1: Status endpoint
test_endpoint \
    "/api/status" \
    "$GHOST_BASE_URL/api/status"

# Test 2: Tick endpoint (first call)
echo "Testing /api/tick (counter increment)..."
tick1=$(curl -s "$GHOST_BASE_URL/api/tick" | grep -oP '"tick":\s*\K\d+' || echo "0")
echo "  First call: tick=$tick1"
sleep 3
tick2=$(curl -s "$GHOST_BASE_URL/api/tick" | grep -oP '"tick":\s*\K\d+' || echo "0")
echo "  Second call: tick=$tick2"
if [ "$tick2" -gt "$tick1" ]; then
    echo -e "  ${GREEN}✓ PASS${NC} - Counter incremented ($tick1 → $tick2)"
    ((passed++))
else
    echo -e "  ${RED}✗ FAIL${NC} - Counter did not increment"
    ((failed++))
fi
echo ""

# Test 3: Regime endpoint
test_endpoint \
    "/api/regime/current" \
    "$GHOST_BASE_URL/api/regime/current"

# Test 4: Price diagnostics (AAPL)
echo "Testing /api/price/diagnostics?symbol=AAPL (routing fix)..."
diag_response=$(curl -s "$GHOST_BASE_URL/api/price/diagnostics?symbol=AAPL" 2>&1)
diag_code=$(curl -s -w '%{http_code}' -o /dev/null "$GHOST_BASE_URL/api/price/diagnostics?symbol=AAPL")

if [[ "$diag_code" =~ ^2 ]]; then
    symbol=$(echo "$diag_response" | grep -oP '"symbol":\s*"\K[^"]+' || echo "UNKNOWN")
    if [ "$symbol" == "AAPL" ]; then
        echo -e "${GREEN}✓ PASS${NC} - Correct symbol routing (HTTP $diag_code, symbol=$symbol)"
        ((passed++))
    else
        echo -e "${RED}✗ FAIL${NC} - Wrong symbol returned (expected AAPL, got $symbol)"
        ((failed++))
    fi
else
    echo -e "${RED}✗ FAIL${NC} (HTTP $diag_code)"
    ((failed++))
fi
echo ""

# Test 5: Cache purge endpoint
test_endpoint \
    "/api/cache/purge (POST)" \
    "$GHOST_BASE_URL/api/cache/purge" \
    "POST" \
    '{"patterns":["price:AAPL","diagnostics:*"]}' \
    "-H 'Authorization: Bearer $GHOST_API_TOKEN' -H 'Content-Type: application/json'"

# Test 6: Movers scanner health
test_endpoint \
    "/api/scan/health" \
    "$GHOST_BASE_URL/api/scan/health"

# Test 7: OpenAPI schema check
echo "Checking OpenAPI schema for new routes..."
openapi_response=$(curl -s "$GHOST_BASE_URL/openapi.json" 2>&1)
if echo "$openapi_response" | grep -q "regime"; then
    echo -e "${GREEN}✓ PASS${NC} - OpenAPI contains /api/regime/current"
    ((passed++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} - OpenAPI may not contain all routes"
fi
echo ""

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "Passed: ${GREEN}$passed${NC}"
echo -e "Failed: ${RED}$failed${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Open in browser: $GHOST_BASE_URL/api/cockpit/stream"
    echo "   Verify SSE events: status, ping, snapshot"
    echo "2. Monitor Railway logs for 5 minutes"
    echo "   Check for: 0×499, 0×502 errors"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "- Check Railway deployment logs"
    echo "- Verify healthcheck passed"
    echo "- Review middleware error handling"
    exit 1
fi
