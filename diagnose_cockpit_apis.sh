#!/bin/bash
BASE_URL="https://ghost-protocol-production.up.railway.app"
echo "========================================"
echo "COCKPIT API ENDPOINT DIAGNOSTICS"
echo "========================================"
echo ""

echo "1. Testing /api/v3/movers (Top Movers):"
curl -s --max-time 5 "$BASE_URL/api/v3/movers?limit=5" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Stocks: {len(d.get('stocks',[]))}, Crypto: {len(d.get('crypto',[]))}\")" 2>/dev/null || echo "  ❌ FAILED"

echo ""
echo "2. Testing /api/vip/coins (VIP Coins):"
curl -s --max-time 5 "$BASE_URL/api/vip/coins" -H "Authorization: Bearer ${GHOST_API_TOKEN:-edaa4eac-6455-4693-a745-142cb6deef03}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Coins: {len(d.get('coins',[]))}\")" 2>/dev/null || echo "  ❌ FAILED or AUTH REQUIRED"

echo ""
echo "3. Testing /api/v3/predictions/latest (Forecast):"
curl -s --max-time 5 "$BASE_URL/api/v3/predictions/latest?symbol=BTC&limit=3" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Predictions: {len(d.get('predictions',[]))}\")" 2>/dev/null || echo "  ❌ FAILED"

echo ""
echo "4. Testing /api/news/latest (News Feed):"
curl -s --max-time 5 "$BASE_URL/api/news/latest?limit=5" | python3 -c "import sys,json; print(json.load(sys.stdin).get('count', 'error'))" 2>/dev/null || echo "  ❌ FAILED"

echo ""
echo "5. Testing /api/v3/watchlist/enriched (Watchlist):"
curl -s --max-time 5 "$BASE_URL/api/v3/watchlist/enriched" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Symbols: {len(d.get('symbols',[]))}\")" 2>/dev/null || echo "  ❌ FAILED"

echo ""
echo "6. Testing /api/v3/goals/snapshot (Goals):"
curl -s --max-time 5 "$BASE_URL/api/v3/goals/snapshot" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Daily: {d.get('daily_goal','N/A')}, Weekly: {d.get('weekly_goal','N/A')}\")" 2>/dev/null || echo "  ❌ FAILED"

echo ""
echo "7. Testing /api/status (Health Score):"
curl -s --max-time 5 "$BASE_URL/api/status" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Score: {d.get('ghost_score','N/A')}\")" 2>/dev/null || echo "  ❌ FAILED"

echo ""
echo "8. Testing /api/cockpit/v3/heartbeat (SSE/Timer):"
curl -s --max-time 2 "$BASE_URL/api/cockpit/v3/heartbeat" | head -3

echo ""
echo "========================================"
