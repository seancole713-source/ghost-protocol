#!/bin/bash
# 🚀 GHOST OMNIBRAIN v10.3 - FULL SYSTEM TEST
# Tests all features: Health, Stocks, Crypto, Telegram, Metrics, Cockpit

set -e

BASE_URL="${1:-http://localhost:5001}"
echo "🧪 Testing GHOST at: $BASE_URL"
echo "========================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

test_endpoint() {
    local name="$1"
    local url="$2"
    local expected="$3"
    
    echo -n "Testing $name... "
    
    response=$(curl -s "$BASE_URL$url" || echo "ERROR")
    
    if echo "$response" | grep -q "$expected"; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}❌ FAIL${NC}"
        echo "  Expected: $expected"
        echo "  Got: $response" | head -c 200
        ((FAILED++))
    fi
}

echo ""
echo "🏥 GROUP A: HEALTH & OBSERVABILITY"
echo "========================================"
test_endpoint "Health Check" "/health" '"ok":true'
test_endpoint "Metrics Export" "/metrics" "ghost_up"
test_endpoint "Metrics - Crypto" "/metrics" "ghost_crypto_price_fetch_total"

echo ""
echo "💰 GROUP B: CRYPTO API"
echo "========================================"
test_endpoint "Crypto Price BTC" "/api/crypto/price/BTC" '"price"'
test_endpoint "Crypto Price ETH" "/api/crypto/price/ETH" '"price"'
test_endpoint "Crypto Price SOL" "/api/crypto/price/SOL" '"price"'
test_endpoint "Crypto Watchlist" "/api/crypto/watchlist" '"assets"'

echo ""
echo "📈 GROUP C: PREDICTIONS"
echo "========================================"
test_endpoint "Stock Prediction" "/api/predict/series?symbol=WOLF" '"path"'

# Crypto prediction - use POST
echo -n "Testing Crypto Prediction Run... "
response=$(curl -s -X POST "$BASE_URL/api/crypto/predict/run?symbol=BTC" || echo "ERROR")
if echo "$response" | grep -q '"forecast_h"'; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAIL${NC}"
    echo "  Got: $response" | head -c 200
    ((FAILED++))
fi

test_endpoint "Crypto Prediction Get" "/api/crypto/predict/BTC" '"path"'

echo ""
echo "🎛️ GROUP D: COCKPIT UI"
echo "========================================"
test_endpoint "Cockpit Data" "/api/cockpit" '"predictions"'
test_endpoint "Cockpit HTML" "/cockpit" "<html"
test_endpoint "Cockpit - Crypto Data" "/api/cockpit" '"crypto"'

echo ""
echo "📱 GROUP E: TELEGRAM (if enabled)"
echo "========================================"
test_endpoint "Telegram Test Endpoint" "/api/telegram/test" '"ok"'

echo ""
echo "💾 GROUP F: DATABASE"
echo "========================================"
test_endpoint "Portfolio Endpoint" "/api/portfolio" '"quantity"'
test_endpoint "Snapshot Endpoint" "/api/snapshot" '"predictions"'

echo ""
echo "========================================"
echo "📊 TEST SUMMARY"
echo "========================================"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED!${NC}"
    echo "GHOST OmniBrain v10.3 is production ready! 🚀"
    exit 0
else
    echo -e "\n${YELLOW}⚠️  Some tests failed${NC}"
    echo "Review failures above and check:"
    echo "  - CRYPTO_ENABLED=1 is set"
    echo "  - Database exists at data/wolf.db"
    echo "  - TELEGRAM_BOT_TOKEN is configured (if testing Telegram)"
    exit 1
fi
