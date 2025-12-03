#!/bin/bash
# Emergency fallback if Railway deployment keeps failing

BASE_URL="https://ghost-protocol-production.up.railway.app"

echo "=========================================="
echo "🚨 EMERGENCY FALLBACK DIAGNOSTICS"
echo "=========================================="
echo ""

# Check if server is even responding
echo "1️⃣  Testing if server is alive..."
if curl -sS --max-time 5 "$BASE_URL/health" > /dev/null 2>&1; then
    echo "✅ Server responding"
else
    echo "❌ Server not responding - Railway deployment failed"
    echo ""
    echo "NEXT STEPS:"
    echo "  1. Check Railway dashboard for deployment logs"
    echo "  2. Look for error before 'Starting Container'"
    echo "  3. Common issues:"
    echo "     - PostgreSQL connection timeout"
    echo "     - Missing environment variables"
    echo "     - Import errors during startup"
    echo ""
    echo "TEMPORARY FIX: Disable personal watchlist scheduler"
    echo "  Run: ./scripts/disable_watchlist_scheduler.sh"
    exit 1
fi

# Check if /health returns valid JSON
echo "2️⃣  Testing /health endpoint..."
health_response=$(curl -sS --max-time 10 "$BASE_URL/health" 2>&1)
if echo "$health_response" | python3 -m json.tool > /dev/null 2>&1; then
    echo "✅ /health returns valid JSON"
    echo "$health_response" | python3 -m json.tool | head -10
else
    echo "⚠️  /health not returning JSON:"
    echo "$health_response" | head -5
fi

echo ""
echo "3️⃣  Testing watchlist endpoints..."
curl -sS --max-time 10 "$BASE_URL/api/v3/watchlist/enriched" | python3 -c "import sys, json; d=json.load(sys.stdin); print('✅ enriched OK' if d.get('ok') else '❌ enriched failed')"
curl -sS --max-time 10 "$BASE_URL/api/v3/watchlist/user" | python3 -c "import sys, json; d=json.load(sys.stdin); print('✅ user OK' if d.get('ok') else '❌ user 404')" 2>&1 | grep -q "404" && echo "❌ /user still 404"

echo ""
echo "4️⃣  Testing recent_alerts..."
alerts_response=$(curl -sS --max-time 10 "$BASE_URL/api/recent_alerts?limit=3" 2>&1)
if echo "$alerts_response" | grep -q "unauthorized"; then
    echo "❌ recent_alerts still requires auth"
    echo "   FIX: Add to public_paths in wolf_app.py"
elif echo "$alerts_response" | grep -q '"ok"'; then
    echo "✅ recent_alerts working"
else
    echo "⚠️  recent_alerts unexpected response:"
    echo "$alerts_response" | head -3
fi

echo ""
echo "=========================================="
echo "🔍 DIAGNOSIS COMPLETE"
echo "=========================================="
