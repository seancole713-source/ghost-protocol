#!/bin/bash
# Test script for Phase Upgrade → 90% Ops

BASE_URL="http://127.0.0.1:8444"

echo "=== Ghost Cockpit Endpoint Tests ==="
echo

# Test 1: Tick
echo "[1/9] Testing /api/tick"
curl -s "$BASE_URL/api/tick"
echo -e "\n"

# Test 2: Regime
echo "[2/9] Testing /api/regime/current"
curl -s "$BASE_URL/api/regime/current"
echo -e "\n"

# Test 3: Goals
echo "[3/9] Testing /api/goals"
curl -s "$BASE_URL/api/goals"
echo -e "\n"

# Test 4: Ghost Score
echo "[4/9] Testing /api/ghost/score"
curl -s "$BASE_URL/api/ghost/score"
echo -e "\n"

# Test 5: News Trending
echo "[5/9] Testing /api/news/trending"
curl -s "$BASE_URL/api/news/trending"
echo -e "\n"

# Test 6: Telegram Test (no auth for now)
echo "[6/9] Testing /api/alerts/test"
curl -s -X POST "$BASE_URL/api/alerts/test"
echo -e "\n"

# Test 7: Price diagnostics for WOLF
echo "[7/9] Testing /api/price/diagnostics?symbol=WOLF"
curl -s "$BASE_URL/api/price/diagnostics?symbol=WOLF"
echo -e "\n"

# Test 8: Price diagnostics for AAPL (should NOT return WOLF price)
echo "[8/9] Testing /api/price/diagnostics?symbol=AAPL"
curl -s "$BASE_URL/api/price/diagnostics?symbol=AAPL"
echo -e "\n"

# Test 9: SSE Stream (sample 10 lines)
echo "[9/9] Testing /api/cockpit/stream (first 10 lines)"
timeout 5 curl -sN "$BASE_URL/api/cockpit/stream" | head -n 10
echo -e "\n"

echo "=== Test Complete ==="
