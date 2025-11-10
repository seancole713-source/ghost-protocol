#!/bin/bash
#
# Test Ghost AI Advisor - Verify all endpoints work
#

set -e

BASE_URL="${1:-http://localhost:8444}"

echo "🧪 Testing Ghost AI Advisor"
echo "Base URL: $BASE_URL"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

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
        if [ -n "$body" ]; then
            echo "   Response preview: $(echo "$body" | head -c 150 | jq -c . 2>/dev/null || echo "$body" | head -c 150)..."
        fi
    elif [ "$http_code" = "503" ]; then
        echo -e "${YELLOW}⚠️  SKIP${NC} (HTTP $http_code - Service not enabled)"
    else
        echo -e "${RED}❌ FAIL${NC} (HTTP $http_code)"
        echo "   Response: $(echo "$body" | head -c 200)"
    fi
    echo ""
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. CRYPTO PHASE 1 ENDPOINTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_endpoint "GET" "/api/crypto/accuracy" "Crypto accuracy tracking"
test_endpoint "GET" "/api/crypto/movers?threshold=10&limit=5" "Crypto top movers"
test_endpoint "GET" "/api/crypto/news?limit=5" "Crypto news feed"
test_endpoint "GET" "/api/crypto/regime/current" "Crypto market regime"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. AI ADVISOR ENDPOINTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_endpoint "POST" "/api/advisor/start" "Start AI advisor"
sleep 2  # Give scanner time to start

test_endpoint "POST" "/api/advisor/scan_now" "Trigger immediate scan"
sleep 5  # Give scan time to complete

test_endpoint "GET" "/api/advisor/recommendations?limit=5" "Get AI recommendations"
test_endpoint "GET" "/api/advisor/stats" "Get AI advisor stats"
test_endpoint "POST" "/api/advisor/stop" "Stop AI advisor"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ If all tests PASS: Ghost AI Advisor is working!"
echo "⚠️  If tests SKIP: Enable modules (CRYPTO_ENABLED=1, AGENTS_ENABLED=1)"
echo "❌ If tests FAIL: Check server logs for errors"
echo ""
echo "Next steps:"
echo "1. Leave AI advisor running (it scans every 30s)"
echo "2. Check recommendations: curl \"$BASE_URL/api/advisor/recommendations\""
echo "3. Monitor accuracy: curl \"$BASE_URL/api/advisor/stats\""
echo ""
