#!/bin/bash
#
# Test script for new crypto endpoints
# Usage: ./test_crypto_endpoints.sh [BASE_URL]
#
# Example:
#   ./test_crypto_endpoints.sh http://localhost:8444
#   ./test_crypto_endpoints.sh https://web-production-8e9a0.up.railway.app
#

set -e

BASE_URL="${1:-http://localhost:${PORT:-8080}}"

echo "🧪 Testing Crypto Endpoints"
echo "Base URL: $BASE_URL"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_endpoint() {
    local method="$1"
    local endpoint="$2"
    local description="$3"
    
    echo -n "Testing $description... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL$endpoint")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✅ PASS${NC} (HTTP $http_code)"
        echo "   Response: $(echo "$body" | head -c 100)..."
    elif [ "$http_code" = "503" ]; then
        echo -e "${YELLOW}⚠️  SKIP${NC} (HTTP $http_code - Module not enabled)"
    else
        echo -e "${RED}❌ FAIL${NC} (HTTP $http_code)"
        echo "   Response: $body"
    fi
    echo ""
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. ACCURACY TRACKING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "GET" "/api/crypto/accuracy" "Accuracy (all symbols)"
test_endpoint "GET" "/api/crypto/accuracy?symbol=BTC" "Accuracy (BTC only)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. TOP MOVERS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "GET" "/api/crypto/movers" "Movers (default threshold)"
test_endpoint "GET" "/api/crypto/movers?threshold=5&limit=10" "Movers (5% threshold, 10 results)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. NEWS FEED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "GET" "/api/crypto/news?limit=10" "News (all cryptos)"
test_endpoint "GET" "/api/crypto/news?symbol=BTC&limit=5" "News (BTC only)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. MARKET REGIME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "GET" "/api/crypto/regime/current" "Regime detection"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. DECISION HISTORY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "GET" "/api/crypto/decisions" "Decisions (all symbols)"
test_endpoint "GET" "/api/crypto/decisions?symbol=ETH&limit=5" "Decisions (ETH only)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. OHLCV TIMESERIES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "GET" "/api/crypto/ohlcv/BTC?days=30&interval=1h" "OHLCV (BTC 30d 1h)"
test_endpoint "GET" "/api/crypto/ohlcv/ETH?days=90&interval=1d" "OHLCV (ETH 90d 1d)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. AI DECISION ENGINE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  Note: This test requires AGENTS_ENABLED=1 and a valid AI provider"
echo "   First, we need to generate a prediction..."

# Generate prediction first
echo -n "   Generating prediction for ETH... "
pred_response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/crypto/predict/run?symbol=ETH")
pred_http_code=$(echo "$pred_response" | tail -n1)

if [ "$pred_http_code" = "200" ]; then
    echo -e "${GREEN}✅${NC}"
    echo ""
    test_endpoint "POST" "/api/crypto/decide?symbol=ETH" "AI decision (ETH)"
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (HTTP $pred_http_code - Prediction failed)"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ If all tests show 200 OK: Endpoints are working"
echo "⚠️  If tests show 503: Module not enabled (need CRYPTO_ENABLED=1)"
echo "❌ If tests show other errors: Check logs for details"
echo ""
echo "To enable crypto features:"
echo "  export CRYPTO_ENABLED=1"
echo "  export AGENTS_ENABLED=1  # For AI decisions"
echo "  export AI_PROVIDER=openai"
echo "  export OPENAI_API_KEY=sk-..."
echo ""
