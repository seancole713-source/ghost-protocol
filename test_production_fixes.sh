#!/bin/bash
# Test production predictions after fix deployment

echo "🧪 Testing Ghost Protocol Production (post-fix)"
echo "================================================"
echo ""

echo "Test 1: BTC (crypto baseline)"
echo "------------------------------"
curl -X POST "https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=BTC" \
  --max-time 10 -s | jq '{ok, symbol, confidence, current_price, duration_ms}'
echo ""

echo "Test 2: PACS (stock - the critical fix)"
echo "----------------------------------------"
curl -X POST "https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=PACS" \
  --max-time 10 -s | jq '{ok, symbol, direction, confidence, current_price, duration_ms, error}'
echo ""

echo "Test 3: AAPL (another stock)"
echo "----------------------------"
curl -X POST "https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=AAPL" \
  --max-time 10 -s | jq '{ok, symbol, confidence, current_price, duration_ms, error}'
echo ""

echo "================================================"
echo "✅ Tests complete"
