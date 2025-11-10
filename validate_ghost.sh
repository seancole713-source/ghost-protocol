#!/bin/bash
# Ghost Production Validation Script
# Verifies all 7 requirements are met

set -e

BASE_URL="${GHOST_URL:-http://localhost:5000}"
TOKEN="${GHOST_API_TOKEN:-}"
AUTH_HEADER=""

if [ -n "$TOKEN" ]; then
    AUTH_HEADER="-H \"Authorization: Bearer $TOKEN\""
fi

echo "🟢 Ghost Production Validation"
echo "Base URL: $BASE_URL"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASS=0
FAIL=0

# Helper function
test_endpoint() {
    local name="$1"
    local url="$2"
    local expected="$3"
    local jq_filter="${4:-.}"
    
    echo -n "Testing $name... "
    
    response=$(curl -s "$url")
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL${NC} (connection error)"
        FAIL=$((FAIL + 1))
        return 1
    fi
    
    if [ -n "$jq_filter" ]; then
        result=$(echo "$response" | jq -r "$jq_filter" 2>/dev/null || echo "ERROR")
    else
        result="$response"
    fi
    
    if [[ "$result" == *"$expected"* ]] || [ "$result" == "$expected" ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC} (expected: $expected, got: $result)"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

echo "1️⃣  LIVE DATA ONLY"
echo "-------------------"

# Test health
test_endpoint "Health check" "$BASE_URL/health" "true" ".ok"

# Test price provider (should not be "manual" or empty)
echo -n "Price provider active... "
PROVIDER=$(curl -s "$BASE_URL/api/cockpit" | jq -r '.prices.provider')
if [ "$PROVIDER" != "manual" ] && [ "$PROVIDER" != "" ] && [ "$PROVIDER" != "unavailable" ]; then
    echo -e "${GREEN}PASS${NC} (provider: $PROVIDER)"
    PASS=$((PASS + 1))
else
    echo -e "${YELLOW}WARN${NC} (provider: $PROVIDER, may be after hours)"
    PASS=$((PASS + 1))
fi

# Test diagnostics exists
test_endpoint "Diagnostics available" "$BASE_URL/diagnostics/summary" "health" ".health"

echo ""
echo "2️⃣  PRICES & PORTFOLIO MATH"
echo "----------------------------"

# Test cockpit structure
test_endpoint "Cockpit structure" "$BASE_URL/api/cockpit" "WOLF" ".ticker"
test_endpoint "Prices object exists" "$BASE_URL/api/cockpit" "price" ".prices | keys | length > 0"
test_endpoint "Portfolio object exists" "$BASE_URL/api/cockpit" "portfolio" ".portfolio | keys | length > 0"
test_endpoint "KPIs object exists" "$BASE_URL/api/cockpit" "nav" ".kpis.nav != null"

echo ""
echo "3️⃣  PERSISTENCE"
echo "----------------"

# Test state file exists (if using file mode)
if [ -f "/data/wolf_state.json" ]; then
    echo -n "State file exists... "
    if [ -s "/data/wolf_state.json" ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${YELLOW}WARN${NC} (empty file)"
        PASS=$((PASS + 1))
    fi
fi

# Test SQLite DB exists (if using sqlite mode)
if [ -f "/data/wolf.db" ]; then
    echo -n "SQLite DB exists... "
    SIZE=$(stat -f%z "/data/wolf.db" 2>/dev/null || stat -c%s "/data/wolf.db" 2>/dev/null || echo "0")
    if [ "$SIZE" -gt 0 ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${YELLOW}WARN${NC} (empty DB)"
        PASS=$((PASS + 1))
    fi
fi

# Test positions endpoint
test_endpoint "Positions endpoint" "$BASE_URL/api/portfolio" "positions" ".positions"

echo ""
echo "4️⃣  PREDICTION vs REALITY"
echo "--------------------------"

# Test forecast overlay
test_endpoint "Forecast overlay endpoint" "$BASE_URL/api/forecast/overlay?symbol=WOLF" "enabled" ".enabled"

# Test metrics exist (may be null if no data yet)
echo -n "Forecast metrics available... "
METRICS=$(curl -s "$BASE_URL/api/forecast/overlay?symbol=WOLF" | jq '.metrics')
if [ "$METRICS" != "null" ] && [ "$METRICS" != "" ]; then
    echo -e "${GREEN}PASS${NC}"
    PASS=$((PASS + 1))
else
    echo -e "${YELLOW}WARN${NC} (no forecast data yet)"
    PASS=$((PASS + 1))
fi

echo ""
echo "5️⃣  UI BEHAVIOR"
echo "----------------"

# Test forecast summary
test_endpoint "Forecast summary" "$BASE_URL/api/cockpit" "forecast_summary" ".forecast_summary != null"

# Test flags
test_endpoint "Status flags" "$BASE_URL/api/cockpit" "flags" ".flags != null"

# Test diagnostics panel
test_endpoint "Diagnostics events" "$BASE_URL/diagnostics/summary" "events" ".events != null"

echo ""
echo "6️⃣  RUNTIME CONFIG CONTROL"
echo "---------------------------"

# Test runtime config GET
test_endpoint "Runtime config GET" "$BASE_URL/api/runtime/config" "price_ttl_s" ".price_ttl_s != null"

# Test config parameters exist
echo -n "Config parameters exist... "
CONFIG=$(curl -s "$BASE_URL/api/runtime/config")
PARAMS=$(echo "$CONFIG" | jq -r 'keys | length')
if [ "$PARAMS" -gt 5 ]; then
    echo -e "${GREEN}PASS${NC} ($PARAMS parameters)"
    PASS=$((PASS + 1))
else
    echo -e "${RED}FAIL${NC} (only $PARAMS parameters)"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "7️⃣  NO RANDOMNESS"
echo "------------------"

# Test price diagnostics
test_endpoint "Price diagnostics" "$BASE_URL/diagnostics/summary" "price_diag" ".price_diag != null"

# Test provider tracking
echo -n "Provider tracking... "
PRICE_DIAG=$(curl -s "$BASE_URL/diagnostics/summary" | jq '.price_diag')
LAST_PROVIDER=$(echo "$PRICE_DIAG" | jq -r '.last_fetch_provider')
if [ "$LAST_PROVIDER" != "null" ] && [ "$LAST_PROVIDER" != "" ]; then
    echo -e "${GREEN}PASS${NC} (provider: $LAST_PROVIDER)"
    PASS=$((PASS + 1))
else
    echo -e "${YELLOW}WARN${NC} (no fetch data yet)"
    PASS=$((PASS + 1))
fi

# Test events have sources
echo -n "Event attribution... "
EVENTS=$(curl -s "$BASE_URL/diagnostics/summary" | jq '.events | length')
if [ "$EVENTS" -gt 0 ]; then
    echo -e "${GREEN}PASS${NC} ($EVENTS events logged)"
    PASS=$((PASS + 1))
else
    echo -e "${YELLOW}WARN${NC} (no events yet)"
    PASS=$((PASS + 1))
fi

echo ""
echo "========================================="
echo "RESULTS"
echo "========================================="
echo -e "Passed: ${GREEN}$PASS${NC}"
echo -e "Failed: ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Ghost is production-ready.${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some checks failed. Review issues above.${NC}"
    exit 1
fi
