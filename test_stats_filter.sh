#!/bin/bash
# Test the new ?since parameter for filtering stats

echo "📊 TESTING STATS FILTER PARAMETER"
echo "=================================="
echo ""

echo "1️⃣  ALL STATS (default - last 30 days):"
echo "   GET /api/v3/paper/stats"
curl -s "https://ghost-protocol-production.up.railway.app/api/v3/paper/stats" | python3 -c "
import sys, json
data = json.load(sys.stdin)
stats = data.get('stats', {})
print(f\"   Total: {stats.get('total_trades', 0):,} trades\")
print(f\"   Wins: {stats.get('wins', 0):,} / Losses: {stats.get('losses', 0):,}\")
print(f\"   Win Rate: {stats.get('win_rate', 0)*100:.1f}%\")
"

echo ""
echo "2️⃣  V2 ERA ONLY (since 2026-01-14):"
echo "   GET /api/v3/paper/stats?since=2026-01-14"
curl -s "https://ghost-protocol-production.up.railway.app/api/v3/paper/stats?since=2026-01-14" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if not data.get('ok'):
    print(f\"   ❌ Error: {data.get('error', 'Unknown')}\")
else:
    stats = data.get('stats', {})
    filters = data.get('filters', {})
    print(f\"   Filter: since={filters.get('since', 'N/A')}\")
    print(f\"   Total: {stats.get('total_trades', 0):,} trades\")
    print(f\"   Wins: {stats.get('wins', 0):,} / Losses: {stats.get('losses', 0):,}\")
    print(f\"   Win Rate: {stats.get('win_rate', 0)*100:.1f}%\")
    
    # V2 symbols
    acc = stats.get('accuracy_by_symbol', {})
    v2_symbols = ['CHZ', 'ZEC', 'RNDR', 'ILV', 'T', 'TURBO', 'RLC', 'EGLD', 'LRC', 'ICP', 'OCEAN']
    v2_wins = 0
    v2_trades = 0
    for sym in v2_symbols:
        if sym in acc:
            v2_trades += acc[sym].get('trades', 0)
            v2_wins += acc[sym].get('wins', 0)
    
    if v2_trades > 0:
        print(f\"   V2 Whitelisted: {v2_wins}/{v2_trades} = {(v2_wins/v2_trades)*100:.1f}%\")
"

echo ""
echo "3️⃣  COMPARISON:"
echo "   Without filter: Shows ALL historical trades (including pre-V2 junk)"
echo "   With since=2026-01-14: Shows only V2 filter era (clean data)"
echo ""
echo "✅ This filter lets you see TRUE V2 performance!"
