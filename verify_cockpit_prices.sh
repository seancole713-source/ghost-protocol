#!/bin/bash

# Cockpit V3 Post-Deployment Verification Script
# Tests that price hydration fix resolved the -- -- display issue

BASE_URL="${RAILWAY_URL:-https://ghost-protocol-production.up.railway.app}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 COCKPIT V3 PRICE HYDRATION FIX - VERIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing: $BASE_URL"
echo "Date: $(date)"
echo ""

echo "Testing /api/v3/watchlist/enriched (should return real prices, not null)..."
response=$(curl -s "$BASE_URL/api/v3/watchlist/enriched")

# Check if we got valid JSON
if ! echo "$response" | python3 -m json.tool > /dev/null 2>&1; then
    echo "❌ FAIL: Invalid JSON response"
    echo "$response"
    exit 1
fi

# Count null prices
null_count=$(echo "$response" | grep -o '"price":null' | wc -l)
total_items=$(echo "$response" | grep -o '"symbol":' | wc -l)

echo ""
echo "Results:"
echo "  Total symbols: $total_items"
echo "  Null prices: $null_count"
echo ""

if [ "$null_count" -eq 0 ]; then
    echo "✅ SUCCESS: All prices populated!"
    echo ""
    echo "Sample prices:"
    echo "$response" | grep -A1 '"symbol":"BTC"' | head -n3
    echo "$response" | grep -A1 '"symbol":"ETH"' | head -n3
    echo "$response" | grep -A1 '"symbol":"AAPL"' | head -n3
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎉 COCKPIT SHOULD NOW DISPLAY PRICES"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Visual verification checklist:"
    echo "  1. Open $BASE_URL/cockpit"
    echo "  2. Check BTC/ETH Major Caps - should show prices (not -- --)"
    echo "  3. Check Watchlist table - should show prices in left column"
    echo "  4. Check Forecast panel - should show Current: \$X and Target: \$Y"
    echo "  5. Verify XRP price in VIP tracker"
    echo ""
    exit 0
elif [ "$null_count" -lt "$total_items" ]; then
    echo "⚠️  PARTIAL: Some prices populated, some still null"
    echo ""
    echo "Null prices found for:"
    echo "$response" | grep -B1 '"price":null' | grep '"symbol"' | head -n5
    echo ""
    echo "This may be expected if some symbols are temporarily offline."
    exit 0
else
    echo "❌ FAIL: All prices still null"
    echo ""
    echo "Possible causes:"
    echo "  1. Fix not yet deployed to Railway"
    echo "  2. Polygon/CoinGecko API keys missing"
    echo "  3. ensure_price_cached() function not working"
    echo ""
    echo "Check Railway deployment logs for errors."
    exit 1
fi
