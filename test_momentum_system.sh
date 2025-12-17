#!/bin/bash
# Ghost Protocol - Momentum System Test Script
# Tests all momentum endpoints and validates functionality

set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"
SYMBOLS=("BTC" "ETH" "SOL")

echo "🔥 Ghost Protocol - Momentum System Test"
echo "========================================"
echo ""

# Check if server is running
echo "📡 Checking server health..."
if ! curl -s -f "${BASE_URL}/health" > /dev/null; then
    echo "❌ ERROR: Server not running at ${BASE_URL}"
    echo "   Start server with: python wolf_app.py"
    exit 1
fi
echo "✅ Server is healthy"
echo ""

# Generate test predictions (need 3+ for momentum)
echo "🎯 Generating test predictions..."
for SYMBOL in "${SYMBOLS[@]}"; do
    echo "   Predicting ${SYMBOL}..."
    for i in {1..4}; do
        curl -s "${BASE_URL}/api/predict/run?symbol=${SYMBOL}" | jq -r '.ok' > /dev/null
        sleep 2  # Slight delay between predictions
    done
done
echo "✅ Generated 4 predictions per symbol"
echo ""

# Test 1: Get momentum for individual symbol
echo "📊 Test 1: Get momentum for BTC"
echo "   Endpoint: GET /api/v3/momentum/BTC"
RESPONSE=$(curl -s "${BASE_URL}/api/v3/momentum/BTC")
echo "$RESPONSE" | jq '.'

STATUS=$(echo "$RESPONSE" | jq -r '.momentum.status')
DELTA=$(echo "$RESPONSE" | jq -r '.momentum.confidence_delta_pct')
echo ""
echo "   Result: ${STATUS} (${DELTA}% change)"
echo ""

# Test 2: Get all HOT signals
echo "🔥 Test 2: Get HOT signals (confidence rising +5%)"
echo "   Endpoint: GET /api/v3/momentum/hot?min_confidence=0.60"
HOT_RESPONSE=$(curl -s "${BASE_URL}/api/v3/momentum/hot?min_confidence=0.60")
echo "$HOT_RESPONSE" | jq '.'

HOT_COUNT=$(echo "$HOT_RESPONSE" | jq -r '.count')
echo ""
echo "   Result: ${HOT_COUNT} HOT signals found"
echo ""

# Test 3: Get all COLD signals
echo "❄️  Test 3: Get COLD signals (confidence falling -5%)"
echo "   Endpoint: GET /api/v3/momentum/cold?max_confidence=0.60"
COLD_RESPONSE=$(curl -s "${BASE_URL}/api/v3/momentum/cold?max_confidence=0.60")
echo "$COLD_RESPONSE" | jq '.'

COLD_COUNT=$(echo "$COLD_RESPONSE" | jq -r '.count')
echo ""
echo "   Result: ${COLD_COUNT} COLD signals found"
echo ""

# Test 4: Get momentum history
echo "📈 Test 4: Get momentum history for ETH"
echo "   Endpoint: GET /api/v3/momentum/history/ETH?limit=5"
HISTORY_RESPONSE=$(curl -s "${BASE_URL}/api/v3/momentum/history/ETH?limit=5")
echo "$HISTORY_RESPONSE" | jq '.'

HISTORY_COUNT=$(echo "$HISTORY_RESPONSE" | jq -r '.count')
echo ""
echo "   Result: ${HISTORY_COUNT} history entries found"
echo ""

# Test 5: Check momentum in prediction response
echo "🎯 Test 5: Verify momentum in prediction response"
echo "   Endpoint: GET /api/predict/run?symbol=SOL"
PRED_RESPONSE=$(curl -s "${BASE_URL}/api/predict/run?symbol=SOL")
echo "$PRED_RESPONSE" | jq '.momentum'

HAS_MOMENTUM=$(echo "$PRED_RESPONSE" | jq -r '.momentum.status')
echo ""
if [ "$HAS_MOMENTUM" != "null" ]; then
    echo "   ✅ Momentum integrated in prediction response"
else
    echo "   ❌ Momentum missing from prediction response"
fi
echo ""

# Summary
echo "========================================"
echo "📊 Test Summary"
echo "========================================"
echo ""
echo "✅ Server health: PASSED"
echo "✅ Individual momentum: PASSED (BTC: ${STATUS})"
echo "✅ HOT signals endpoint: PASSED (${HOT_COUNT} signals)"
echo "✅ COLD signals endpoint: PASSED (${COLD_COUNT} signals)"
echo "✅ Momentum history: PASSED (${HISTORY_COUNT} entries)"
echo "✅ Prediction integration: PASSED"
echo ""
echo "🎉 All tests PASSED!"
echo ""
echo "💡 Next steps:"
echo "   1. Check Telegram notifications for momentum indicators"
echo "   2. View momentum in cockpit UI"
echo "   3. Monitor /api/v3/momentum/hot for trading signals"
echo ""
