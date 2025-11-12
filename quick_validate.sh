#!/usr/bin/env bash
# Quick validation commands after Railway redeploy

export GHOST_BASE_URL="https://ghost-sniper-bot-seancole713-production.up.railway.app"

echo "🔍 Ghost Cockpit Quick Validation"
echo "=================================="
echo ""

echo "1️⃣  Status Check:"
curl -s "$GHOST_BASE_URL/api/status" | python3 -m json.tool
echo ""

echo "2️⃣  Regime Endpoint:"
curl -s "$GHOST_BASE_URL/api/regime/current" | python3 -m json.tool
echo ""

echo "3️⃣  Tick Counter (first):"
TICK1=$(curl -s "$GHOST_BASE_URL/api/tick" | python3 -m json.tool)
echo "$TICK1"
echo ""

echo "⏳ Waiting 5 seconds..."
sleep 5
echo ""

echo "4️⃣  Tick Counter (second - should increase):"
TICK2=$(curl -s "$GHOST_BASE_URL/api/tick" | python3 -m json.tool)
echo "$TICK2"
echo ""

echo "5️⃣  WOLF Diagnostics:"
curl -s "$GHOST_BASE_URL/api/price/diagnostics?symbol=WOLF" | python3 -m json.tool | head -20
echo ""

echo "6️⃣  AAPL Diagnostics:"
curl -s "$GHOST_BASE_URL/api/price/diagnostics?symbol=AAPL" | python3 -m json.tool | head -20
echo ""

echo "7️⃣  Portfolio:"
curl -s "$GHOST_BASE_URL/api/portfolio" | python3 -m json.tool | head -15
echo ""

echo "8️⃣  Position:"
curl -s "$GHOST_BASE_URL/api/position" | python3 -m json.tool
echo ""

echo "9️⃣  Movers Scanner Health:"
curl -s "$GHOST_BASE_URL/api/scan/health" | python3 -m json.tool
echo ""

echo "✅ Validation complete!"
echo ""
echo "📡 Open SSE stream in browser:"
echo "   $GHOST_BASE_URL/api/cockpit/stream"
echo ""
echo "🔧 Purge cache if needed:"
echo "   curl -X POST $GHOST_BASE_URL/api/cache/purge -H 'Content-Type: application/json' -d '{\"keys\":[\"price:AAPL\",\"price:WOLF\"]}'"
