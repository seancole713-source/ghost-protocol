#!/bin/bash
# Ghost Protocol Full System Audit v3
# Tests ALL components with VERIFIED endpoint paths

RAILWAY_URL="https://ghost-protocol-production.up.railway.app"
SECRET="o4Cf0tKYUkjPkjXddbtLcLBZeZ7YnxYc"

PASSED=0
FAILED=0
FAILURES=""

test_endpoint() {
    local name="$1"
    local url="$2"
    local check="$3"
    
    echo -n "Testing: $name... "
    response=$(curl -s --max-time 30 "$url")
    
    if echo "$response" | grep -q "$check"; then
        echo "✅ PASS"
        ((PASSED++))
    else
        echo "❌ FAIL"
        ((FAILED++))
        FAILURES="$FAILURES\n- $name: Expected '$check'"
        echo "   Response: $(echo "$response" | head -c 150)"
    fi
}

test_post_endpoint() {
    local name="$1"
    local url="$2"
    local check="$3"
    
    echo -n "Testing: $name... "
    response=$(curl -s --max-time 60 -X POST "$url" -H "X-Cron-Secret: $SECRET")
    
    if echo "$response" | grep -q "$check"; then
        echo "✅ PASS"
        ((PASSED++))
    else
        echo "❌ FAIL"
        ((FAILED++))
        FAILURES="$FAILURES\n- $name: Expected '$check'"
        echo "   Response: $(echo "$response" | head -c 150)"
    fi
}

echo "========================================"
echo "🔍 GHOST PROTOCOL FULL SYSTEM AUDIT v3"
echo "   $(date)"
echo "========================================"
echo ""

echo "--- 1. CORE HEALTH ---"
test_endpoint "1.1 Health Check" "$RAILWAY_URL/health" '"status":"healthy"'
test_endpoint "1.2 BTC Price" "$RAILWAY_URL/api/crypto/price/BTC" '"symbol":"BTC"'
test_endpoint "1.3 ETH Price" "$RAILWAY_URL/api/crypto/price/ETH" '"symbol":"ETH"'
test_endpoint "1.4 SOL Price" "$RAILWAY_URL/api/crypto/price/SOL" '"symbol":"SOL"'

echo ""
echo "--- 2. PREDICTIONS ENGINE ---"
test_endpoint "2.1 Latest Predictions" "$RAILWAY_URL/api/v3/predictions/latest" '"predictions"'
test_endpoint "2.2 Prediction Health" "$RAILWAY_URL/api/health/predictions" '"ok":true'
test_endpoint "2.3 Watchlist Config" "$RAILWAY_URL/api/predictions/symbols" '"multi_symbol_watchlist"'

echo ""
echo "--- 3. TOP 10 SYSTEM ---"
test_endpoint "3.1 TOP 10 Status" "$RAILWAY_URL/alerts/top10/status?secret=$SECRET" '"ok"'

echo ""
echo "--- 4. WATCHDOG TRACKING (PostgreSQL) ---"
test_endpoint "4.1 PostgreSQL Active" "$RAILWAY_URL/debug/tracking-status?secret=$SECRET" '"postgresql"'
test_endpoint "4.2 Persistent Flag" "$RAILWAY_URL/debug/tracking-status?secret=$SECRET" '"persistent":true'
test_endpoint "4.3 Active Picks Count" "$RAILWAY_URL/debug/tracking-status?secret=$SECRET" '"active_picks"'
test_post_endpoint "4.4 Watchdog Check" "$RAILWAY_URL/alerts/watchdog/check" '"ok":true'

echo ""
echo "--- 5. ACCURACY & RECONCILIATION ---"
test_endpoint "5.1 Accuracy Summary" "$RAILWAY_URL/api/v3/accuracy/summary" '"total_predictions"'
test_endpoint "5.2 Accuracy Dashboard" "$RAILWAY_URL/api/v3/accuracy/dashboard" '"ok"'
test_post_endpoint "5.3 Reconcile Trigger" "$RAILWAY_URL/api/v3/reconcile/trigger" '"ok":true'

echo ""
echo "--- 6. PRICE PROVIDERS ---"
test_endpoint "6.1 Price Has Provider" "$RAILWAY_URL/api/crypto/price/BTC" '"provider"'
test_endpoint "6.2 Price Has Cache" "$RAILWAY_URL/api/crypto/price/BTC" '"cached_at"'

echo ""
echo "========================================"
echo "📊 AUDIT RESULTS"
echo "========================================"
echo "✅ PASSED: $PASSED"
echo "❌ FAILED: $FAILED"
TOTAL=$((PASSED + FAILED))
echo "📈 SCORE: $PASSED/$TOTAL"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "🔴 FAILURES:"
    echo -e "$FAILURES"
else
    echo "🟢 ALL SYSTEMS OPERATIONAL"
fi
echo ""
echo "========================================"
