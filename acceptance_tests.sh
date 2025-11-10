#!/bin/bash
# Ghost Cockpit 100% Ops - Acceptance Tests
# Must all pass for 100% operational status

set -e

BASE_URL="${GHOST_BASE_URL:-http://127.0.0.1:8444}"
TOKEN="${GHOST_API_TOKEN:-}"

echo "=================================================="
echo "Ghost Cockpit 100% Ops - Acceptance Tests"
echo "=================================================="
echo "Base URL: $BASE_URL"
echo

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

check_test() {
    local name="$1"
    local result="$2"
    
    if [ "$result" == "PASS" ]; then
        echo -e "${GREEN}✓ PASS${NC}: $name"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $name - $result"
        ((FAILED++))
    fi
}

# TEST 1: Stock Price - AAPL
echo "[1/10] Testing AAPL Live Price..."
AAPL_RESP=$(curl -s "$BASE_URL/api/price/diagnostics?symbol=AAPL")
AAPL_PRICE=$(echo "$AAPL_RESP" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('price', 0))" 2>/dev/null || echo "0")
AAPL_PROVIDER=$(echo "$AAPL_RESP" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('provider', 'none'))" 2>/dev/null || echo "none")
AAPL_FRESH=$(echo "$AAPL_RESP" | python3 -c "import sys, json; d=json.load(sys.stdin); print('true' if d.get('price') and float(d.get('price', 0)) > 0 else 'false')" 2>/dev/null || echo "false")

if [ "$AAPL_PRICE" != "17.95" ] && [ "$AAPL_PRICE" != "0" ] && [[ "$AAPL_PROVIDER" =~ ^(polygon|alphavantage|yfinance|yahoo)$ ]]; then
    check_test "AAPL price routing" "PASS"
    echo "  Price: \$$AAPL_PRICE, Provider: $AAPL_PROVIDER"
else
    check_test "AAPL price routing" "FAIL - Price: $AAPL_PRICE (expect != 17.95), Provider: $AAPL_PROVIDER (expect polygon|alphavantage|yfinance|yahoo)"
fi
echo

# TEST 2: Crypto Price - BTC
echo "[2/10] Testing BTC Live Price..."
BTC_RESP=$(curl -s "$BASE_URL/api/crypto/price/BTC")
BTC_PRICE=$(echo "$BTC_RESP" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('price', 0))" 2>/dev/null || echo "0")

if [ "$BTC_PRICE" != "0" ] && [ $(echo "$BTC_PRICE > 1000" | bc -l) -eq 1 ]; then
    check_test "BTC live price" "PASS"
    echo "  Price: \$$BTC_PRICE"
else
    check_test "BTC live price" "FAIL - Price: $BTC_PRICE (expect > 1000)"
fi
echo

# TEST 3: Six Endpoints - Non-empty responses
echo "[3/10] Testing Six Required Endpoints..."
ENDPOINTS_PASSED=0
ENDPOINTS_FAILED=0

for endpoint in "tick" "regime/current" "goals" "ghost/score" "news/trending"; do
    RESP=$(curl -s "$BASE_URL/api/$endpoint")
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/$endpoint")
    
    # Check if response is not empty and has 200 status
    if [ "$HTTP_CODE" == "200" ] && [ -n "$RESP" ] && [ "$RESP" != "{}" ]; then
        echo "  ✓ /api/$endpoint: 200 OK"
        ((ENDPOINTS_PASSED++))
    else
        echo "  ✗ /api/$endpoint: HTTP $HTTP_CODE, Response: ${RESP:0:50}"
        ((ENDPOINTS_FAILED++))
    fi
done

if [ $ENDPOINTS_FAILED -eq 0 ]; then
    check_test "Six endpoints non-empty" "PASS"
else
    check_test "Six endpoints non-empty" "FAIL - $ENDPOINTS_FAILED/5 endpoints failed"
fi
echo

# TEST 4: Stock Prediction
echo "[4/10] Testing Stock Prediction (AAPL)..."
if [ -z "$TOKEN" ]; then
    check_test "Stock prediction" "SKIP - No GHOST_API_TOKEN set"
    echo
else
    PRED_RESP=$(curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
        -X POST -d '{"symbol":"AAPL","horizon_h":48}' "$BASE_URL/api/predict/run")
    PRED_OK=$(echo "$PRED_RESP" | python3 -c "import sys, json; d=json.load(sys.stdin); print('true' if d.get('ok') or d.get('prediction_id') else 'false')" 2>/dev/null || echo "false")
    PRED_ERROR=$(echo "$PRED_RESP" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('detail', ''))" 2>/dev/null || echo "")
    
    if [ "$PRED_OK" == "true" ] && [[ ! "$PRED_ERROR" =~ "Unable to fetch" ]]; then
        check_test "Stock prediction (AAPL)" "PASS"
    else
        check_test "Stock prediction (AAPL)" "FAIL - Error: $PRED_ERROR"
    fi
    echo
fi

# TEST 5: Crypto Prediction
echo "[5/10] Testing Crypto Prediction (BTC)..."
if [ -z "$TOKEN" ]; then
    check_test "Crypto prediction" "SKIP - No GHOST_API_TOKEN set"
    echo
else
    CRYPTO_PRED=$(curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
        -X POST -d '{"symbol":"BTC","horizon_h":48}' "$BASE_URL/api/crypto/predict/run")
    CRYPTO_OK=$(echo "$CRYPTO_PRED" | python3 -c "import sys, json; d=json.load(sys.stdin); print('true' if d.get('ok') or d.get('forecast') else 'false')" 2>/dev/null || echo "false")
    
    if [ "$CRYPTO_OK" == "true" ]; then
        check_test "Crypto prediction (BTC)" "PASS"
    else
        check_test "Crypto prediction (BTC)" "FAIL - Response: ${CRYPTO_PRED:0:100}"
    fi
    echo
fi

# TEST 6: SSE Stream Events
echo "[6/10] Testing SSE Stream Events..."
SSE_SAMPLE=$(timeout 10 curl -sN "$BASE_URL/api/cockpit/stream" | head -n 60)

HAS_STATUS=$(echo "$SSE_SAMPLE" | grep -c "event: status" || echo "0")
HAS_PING=$(echo "$SSE_SAMPLE" | grep -c "event: ping" || echo "0")
HAS_SNAPSHOT=$(echo "$SSE_SAMPLE" | grep -c "event: snapshot" || echo "0")

if [ "$HAS_STATUS" -gt 0 ] && [ "$HAS_PING" -gt 0 ] && [ "$HAS_SNAPSHOT" -gt 0 ]; then
    check_test "SSE events (status/ping/snapshot)" "PASS"
    echo "  status events: $HAS_STATUS, ping events: $HAS_PING, snapshot events: $HAS_SNAPSHOT"
else
    check_test "SSE events (status/ping/snapshot)" "FAIL - status: $HAS_STATUS, ping: $HAS_PING, snapshot: $HAS_SNAPSHOT"
fi
echo

# TEST 7: Telegram Test Endpoint
echo "[7/10] Testing Telegram Alert..."
TG_RESP=$(curl -s -X POST "$BASE_URL/api/alerts/test")
TG_OK=$(echo "$TG_RESP" | python3 -c "import sys, json; d=json.load(sys.stdin); print('true' if d.get('ok') else 'false')" 2>/dev/null || echo "false")
TG_MSG_ID=$(echo "$TG_RESP" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('message_id', 'none'))" 2>/dev/null || echo "none")

if [ "$TG_OK" == "true" ] && [ "$TG_MSG_ID" != "none" ]; then
    check_test "Telegram alert test" "PASS"
    echo "  Message ID: $TG_MSG_ID"
else
    check_test "Telegram alert test" "FAIL - Response: ${TG_RESP:0:100}"
fi
echo

# TEST 8: ENV Gates Validation
echo "[8/10] Testing ENV Gates..."
STATUS_RESP=$(curl -s "$BASE_URL/api/status")
MODE=$(echo "$STATUS_RESP" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('mode', 'unknown'))" 2>/dev/null || echo "unknown")

if [ "$MODE" == "live" ]; then
    check_test "ENV gates (SIM_MODE=0)" "PASS"
else
    check_test "ENV gates (SIM_MODE=0)" "FAIL - Mode: $MODE (expect 'live')"
fi
echo

# TEST 9: HTTP Error Check (30 second sample)
echo "[9/10] Testing HTTP Stability (30s sample)..."
ERROR_COUNT=0
for i in {1..6}; do
    STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/portfolio")
    if [ "$STATUS_CODE" == "499" ] || [ "$STATUS_CODE" == "502" ]; then
        ((ERROR_COUNT++))
    fi
    sleep 5
done

if [ $ERROR_COUNT -eq 0 ]; then
    check_test "HTTP stability (0×499/502)" "PASS"
else
    check_test "HTTP stability (0×499/502)" "FAIL - Found $ERROR_COUNT errors in 30s"
fi
echo

# TEST 10: Tick Counter Active
echo "[10/10] Testing Tick Counter..."
TICK1=$(curl -s "$BASE_URL/api/tick" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('tick', 0))" 2>/dev/null || echo "0")
sleep 10
TICK2=$(curl -s "$BASE_URL/api/tick" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('tick', 0))" 2>/dev/null || echo "0")

if [ "$TICK2" -gt "$TICK1" ]; then
    check_test "Tick counter incrementing" "PASS"
    echo "  Tick 1: $TICK1 → Tick 2: $TICK2 (Δ=$(($TICK2-$TICK1)))"
else
    check_test "Tick counter incrementing" "FAIL - Tick not incrementing (T1:$TICK1, T2:$TICK2)"
fi
echo

# Summary
echo "=================================================="
echo "TEST RESULTS"
echo "=================================================="
echo -e "${GREEN}PASSED: $PASSED${NC}"
echo -e "${RED}FAILED: $FAILED${NC}"
echo

OPS_PERCENT=$(echo "scale=1; ($PASSED / 10) * 100" | bc)
echo "Operations %: $OPS_PERCENT%"
echo

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED - 100% OPERATIONAL${NC}"
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED - NOT READY FOR PRODUCTION${NC}"
    exit 1
fi
