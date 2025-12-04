#!/bin/bash
BASE_URL="https://ghost-protocol-production.up.railway.app"

echo "🔍 COCKPIT MONITORING"
echo "===================="
echo ""

# 1. Health check
echo -n "Server Health: "
HEALTH=$(curl -sS --max-time 3 "$BASE_URL/health" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['status']} - Uptime: {d['uptime']}s\")" 2>/dev/null)
echo "${HEALTH:-❌ FAILED}"

# 2. Hunter feed (Top Movers / News)
echo -n "Hunter Feed: "
FEED=$(curl -sS --max-time 3 "$BASE_URL/api/v3/hunter/feed?limit=1" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['count']} predictions available\")" 2>/dev/null)
echo "${FEED:-❌ FAILED}"

# 3. VIP Coins
echo -n "VIP Coins: "
VIP=$(curl -sS --max-time 3 "$BASE_URL/api/v3/vip/snapshot" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); online=[c for c in d.get('vip_coins',[]) if c.get('status')=='online']; print(f\"{len(online)}/{len(d.get('vip_coins',[]))} online\")" 2>/dev/null)
echo "${VIP:-❌ FAILED}"

# 4. Watchlist
echo -n "Watchlist: "
WATCHLIST=$(curl -sS --max-time 3 "$BASE_URL/api/v3/watchlist/enriched" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); items=d.get('items',[]); with_price=[i for i in items if i.get('price') is not None]; print(f\"{len(items)} symbols ({len(with_price)} with prices)\")" 2>/dev/null)
echo "${WATCHLIST:-❌ FAILED}"

echo ""
echo "Population Script Progress:"
if [ -f cockpit_population.log ]; then
    tail -3 cockpit_population.log | grep -E "^\[|SUCCESS|FAILED|TIMEOUT|SUMMARY" | tail -3
else
    echo "  Not running"
fi

echo ""
