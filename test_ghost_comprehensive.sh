#!/bin/bash
#
# Ghost Comprehensive Smoke Test
# ================================
# Tests all critical endpoints and functionality
#

set -e

GHOST_URL="${GHOST_URL:-https://web-production-8e9a0.up.railway.app}"
API_TOKEN="${GHOST_API_TOKEN:-}"
FAILED=0
PASSED=0

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Ghost Comprehensive Smoke Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Target: $GHOST_URL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Helper functions
test_endpoint() {
    local name="$1"
    local url="$2"
    local expected_key="$3"
    
    echo -n "Testing $name... "
    
    response=$(curl -s -w "\n%{http_code}" "$url")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" = "200" ]; then
        if [ -n "$expected_key" ]; then
            if echo "$body" | jq -e "$expected_key" > /dev/null 2>&1; then
                echo "✅ PASS"
                PASSED=$((PASSED + 1))
                return 0
            else
                echo "❌ FAIL (missing key: $expected_key)"
                FAILED=$((FAILED + 1))
                return 1
            fi
        else
            echo "✅ PASS"
            PASSED=$((PASSED + 1))
            return 0
        fi
    else
        echo "❌ FAIL (HTTP $http_code)"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

test_authenticated_endpoint() {
    local name="$1"
    local url="$2"
    local expected_key="$3"
    
    if [ -z "$API_TOKEN" ]; then
        echo "⚠️  Skipping $name (no API token)"
        return 0
    fi
    
    echo -n "Testing $name... "
    
    response=$(curl -s -w "\n%{http_code}" -H "Authorization: Bearer $API_TOKEN" "$url")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" = "200" ]; then
        if [ -n "$expected_key" ]; then
            if echo "$body" | jq -e "$expected_key" > /dev/null 2>&1; then
                echo "✅ PASS"
                PASSED=$((PASSED + 1))
                return 0
            else
                echo "❌ FAIL (missing key: $expected_key)"
                echo "Response: $body"
                FAILED=$((FAILED + 1))
                return 1
            fi
        else
            echo "✅ PASS"
            PASSED=$((PASSED + 1))
            return 0
        fi
    else
        echo "❌ FAIL (HTTP $http_code)"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "📋 1. Core Health Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_endpoint "Health (basic)" "$GHOST_URL/health" ".ok"
test_endpoint "Health (detailed)" "$GHOST_URL/health/detailed" ".components"
test_endpoint "API Version" "$GHOST_URL/api/version" ".version"
test_endpoint "API Config" "$GHOST_URL/api/config" ".ticker"

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "📊 2. Portfolio & Position Endpoints"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_endpoint "Position" "$GHOST_URL/api/position" ".symbol"
test_endpoint "Positions (all)" "$GHOST_URL/api/positions" "."

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "🧠 3. AI Memory Endpoints"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_authenticated_endpoint "AI Memory Stats" "$GHOST_URL/ai/memory/stats" ".count"
test_authenticated_endpoint "AI Recent Decisions" "$GHOST_URL/ai/memory/recent?limit=5" "."

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "📈 4. Market Data Endpoints"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_authenticated_endpoint "Current Price" "$GHOST_URL/api/price/WOLF" ".price"

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "🎯 5. Trading Signal Endpoints"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_endpoint "Forecast Overlay" "$GHOST_URL/api/forecast/overlay" "."

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "🔐 6. Security & Secrets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_endpoint "Secrets Health" "$GHOST_URL/api/secrets/health" ".present"

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "📊 7. Database & Persistence"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test detailed health for database checks
echo -n "Testing Portfolio Persistence... "
response=$(curl -s "$GHOST_URL/health/detailed")
if echo "$response" | jq -e '.components.positions' > /dev/null 2>&1; then
    echo "✅ PASS"
    PASSED=$((PASSED + 1))
else
    echo "❌ FAIL"
    FAILED=$((FAILED + 1))
fi

echo -n "Testing AI Memory Database... "
if echo "$response" | jq -e '.components.ai_memory' > /dev/null 2>&1; then
    records=$(echo "$response" | jq -r '.components.ai_memory.records // 0')
    echo "✅ PASS ($records records)"
    PASSED=$((PASSED + 1))
else
    echo "❌ FAIL"
    FAILED=$((FAILED + 1))
fi

echo -n "Testing Price Cache... "
if echo "$response" | jq -e '.components.cache' > /dev/null 2>&1; then
    cache_size=$(echo "$response" | jq -r '.components.cache.price_cache_size // 0')
    echo "✅ PASS ($cache_size cached prices)"
    PASSED=$((PASSED + 1))
else
    echo "❌ FAIL"
    FAILED=$((FAILED + 1))
fi

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "📱 8. Telegram Bot"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "Testing Telegram Webhook Endpoint... "
# Just check if endpoint exists (returns 200 or 405 method not allowed)
webhook_status=$(curl -s -o /dev/null -w "%{http_code}" "$GHOST_URL/telegram/webhook")
if [ "$webhook_status" = "200" ] || [ "$webhook_status" = "405" ]; then
    echo "✅ PASS (endpoint exists)"
    PASSED=$((PASSED + 1))
else
    echo "❌ FAIL (HTTP $webhook_status)"
    FAILED=$((FAILED + 1))
fi

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "📊 FINAL RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Passed: $PASSED"
echo "❌ Failed: $FAILED"
TOTAL=$((PASSED + FAILED))
echo "📊 Total: $TOTAL tests"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All tests passed!"
    exit 0
else
    PASS_RATE=$((PASSED * 100 / TOTAL))
    echo "⚠️  Some tests failed (${PASS_RATE}% pass rate)"
    exit 1
fi
