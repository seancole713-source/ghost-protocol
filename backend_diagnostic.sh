#!/usr/bin/env bash
###############################################################################
# Ghost Cockpit Backend Diagnostic & Validation
# Verifies stabilization: routes, middleware, timeouts, error rates
###############################################################################

set -euo pipefail

GHOST_BASE_URL="${GHOST_BASE_URL:-https://ghost-sniper-bot-seancole713-production.up.railway.app}"

echo "🔍 Ghost Cockpit Backend Diagnostic"
echo "===================================="
echo "Target: $GHOST_BASE_URL"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
TOTAL_LATENCIES=()

test_endpoint() {
    local name="$1"
    local url="$2"
    local method="${3:-GET}"
    local data="${4:-}"
    
    echo -e "${CYAN}▶ Testing: $name${RESET}"
    
    start_time=$(date +%s%3N)
    if [ "$method" = "POST" ] && [ -n "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "$url" \
            -H "Content-Type: application/json" \
            -d "$data" 2>&1)
    else
        response=$(curl -s -w "\n%{http_code}" "$url" 2>&1)
    fi
    end_time=$(date +%s%3N)
    
    http_code=$(echo "$response" | tail -1)
    body=$(echo "$response" | head -n -1)
    latency=$((end_time - start_time))
    TOTAL_LATENCIES+=($latency)
    
    if [ "$http_code" = "200" ]; then
        echo -e "  ${GREEN}✓${RESET} Status: $http_code (${latency}ms)"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [ "$http_code" = "404" ] || [ "$http_code" = "499" ]; then
        echo -e "  ${RED}✗${RESET} Status: $http_code (CRITICAL ERROR)"
        echo "  Body: $body"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        echo -e "  ${YELLOW}⚠${RESET} Status: $http_code (${latency}ms)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
}

echo "1️⃣  Core Health Checks"
echo "====================="
test_endpoint "API Status" "$GHOST_BASE_URL/api/status"
test_endpoint "OpenAPI Schema" "$GHOST_BASE_URL/openapi.json"

echo "2️⃣  New/Fixed Endpoints"
echo "======================"
test_endpoint "Regime Current" "$GHOST_BASE_URL/api/regime/current"
test_endpoint "Tick Counter" "$GHOST_BASE_URL/api/tick"
test_endpoint "Cache Purge" "$GHOST_BASE_URL/api/cache/purge" "POST" '{"keys":["price:TEST"]}'

echo "3️⃣  Diagnostics Routing"
echo "======================"
test_endpoint "WOLF Diagnostics" "$GHOST_BASE_URL/api/price/diagnostics?symbol=WOLF"
test_endpoint "AAPL Diagnostics" "$GHOST_BASE_URL/api/price/diagnostics?symbol=AAPL"

echo "4️⃣  Core Trading Endpoints"
echo "=========================="
test_endpoint "Portfolio" "$GHOST_BASE_URL/api/portfolio"
test_endpoint "Position" "$GHOST_BASE_URL/api/position"
test_endpoint "Cockpit" "$GHOST_BASE_URL/api/cockpit"

echo "5️⃣  Movers Scanner"
echo "=================="
test_endpoint "Movers Health" "$GHOST_BASE_URL/api/scan/health"
test_endpoint "Movers Data" "$GHOST_BASE_URL/api/scan/movers"

# Calculate average latency
if [ ${#TOTAL_LATENCIES[@]} -gt 0 ]; then
    sum=0
    for lat in "${TOTAL_LATENCIES[@]}"; do
        sum=$((sum + lat))
    done
    avg_latency=$((sum / ${#TOTAL_LATENCIES[@]}))
else
    avg_latency=0
fi

echo ""
echo "📊 Summary"
echo "=========="
echo -e "  Tests Passed: ${GREEN}$PASS_COUNT${RESET}"
echo -e "  Tests Failed: ${RED}$FAIL_COUNT${RESET}"
echo -e "  Avg Latency: ${CYAN}${avg_latency}ms${RESET}"

# Check for critical errors (404/499)
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "\n${RED}⚠ CRITICAL: Found $FAIL_COUNT failed tests${RESET}"
    echo "Review logs and redeploy if needed"
    exit 1
fi

# Check latency threshold (should be < 3000ms)
if [ $avg_latency -gt 3000 ]; then
    echo -e "\n${YELLOW}⚠ WARNING: Average latency ${avg_latency}ms exceeds 3s threshold${RESET}"
    echo "Consider provider timeout optimizations"
fi

echo -e "\n${GREEN}✅ Backend validation complete${RESET}"
exit 0
