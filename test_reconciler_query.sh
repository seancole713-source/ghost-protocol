#!/bin/bash
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Paper Trade Reconciler Query Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test the OLD query (will fail with text comparison)
echo "❌ OLD QUERY (target_time <= NOW()):"
echo "SELECT DISTINCT symbol FROM paper_trades WHERE outcome = 'PENDING' AND target_time <= NOW();"
echo ""
curl -s "https://ghost-protocol-production.up.railway.app/api/v3/paper/trades?limit=5" | python3 -c "
import sys, json
from datetime import datetime
data = json.load(sys.stdin)
trades = data.get('trades', [])
now = datetime.utcnow()

due_symbols = set()
for t in trades:
    if t['outcome'] == 'PENDING':
        target = datetime.fromisoformat(t['target_time'].replace('Z', ''))
        if target <= now:
            due_symbols.add(t['symbol'])

if due_symbols:
    print(f'  Should find: {list(due_symbols)}')
else:
    print('  No trades due yet')
"
echo ""

# Test the NEW query (with type casting)
echo "✅ NEW QUERY (target_time::timestamp <= NOW()):"
echo "SELECT DISTINCT symbol FROM paper_trades WHERE outcome = 'PENDING' AND target_time::timestamp <= NOW();"
echo ""
echo "  After fix is deployed, this should return symbols with due trades"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Current Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s "https://ghost-protocol-production.up.railway.app/api/v3/paper/stats?since=2026-01-14" | python3 -c "
import sys, json
data = json.load(sys.stdin)
stats = data.get('stats', {})
print(f\"  Total trades: {stats.get('total_trades', 0)}\")
print(f\"  Resolved: {stats.get('resolved_trades', 0)}\")
print(f\"  Pending: {stats.get('pending_trades', 0)}\")
print(f\"  Win rate: {stats.get('win_rate', 0):.1%}\")
"
