#!/bin/bash
# ============================================================================
# GHOST PROTOCOL 14-DAY EVALUATION
# Deploy date: Feb 9, 2026
# Evaluation date: Feb 22, 2026
# 
# Decision criteria:
#   ≥60% edge symbol WR → Small real money test
#   50-60% → Rebuild (Option B)
#   <50% → Stop (Option C)
# ============================================================================

API="https://ghost-protocol-production.up.railway.app"

echo "=========================================="
echo " GHOST PROTOCOL — 14-DAY EVALUATION"
echo " $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="
echo ""

# Edge symbols whitelist
EDGE="T,GME,TURBO,RNDR,ENJ,JUP,BAND,HOOD,IQ,BMBL,HBAR,XPO,PEPE,IOTX,GIGA,COIN,ILV,BCH,CHZ,ALICE,YFI,ITRI,ICP,BRETT"

# Pull 14-day stats
echo "Pulling 14-day paper trade stats..."
curl -s "$API/api/v3/paper/stats?days=14" | python3 -c "
import json,sys
d=json.load(sys.stdin)
s=d.get('stats',{})
by_sym = s.get('accuracy_by_symbol',{})

edge = set('$EDGE'.split(','))
edge_w, edge_l, nonedge_w, nonedge_l = 0,0,0,0

print('─' * 60)
print('OVERALL (ALL SYMBOLS):')
print(f'  Total trades:    {s.get(\"total_trades\",0)}')
print(f'  Resolved:        {s.get(\"resolved_trades\",0)}')
print(f'  Wins / Losses:   {s.get(\"wins\",0)}W / {s.get(\"losses\",0)}L')
print(f'  Win Rate:        {s.get(\"win_rate_pct\",0):.1f}%')
print()

print('─' * 60)
print('EDGE SYMBOLS (should be ≥60% to proceed):')
print(f\"  {'Symbol':<8} {'Trades':>7} {'Wins':>5} {'Losses':>7} {'WR':>6}\")
print(f\"  {'─'*8} {'─'*7} {'─'*5} {'─'*7} {'─'*6}\")

for sym in sorted(edge):
    data = by_sym.get(sym, {})
    w = data.get('wins',0)
    l = data.get('losses',0)
    t = data.get('trades',0)
    wr = data.get('win_rate',0)
    edge_w += w
    edge_l += l
    if t > 0:
        marker = '✅' if wr >= 0.6 else '⚠️' if wr >= 0.5 else '❌'
        print(f'  {sym:<8} {t:>7} {w:>5} {l:>7} {wr:>5.0%} {marker}')

for sym, data in by_sym.items():
    if sym.upper() not in edge:
        nonedge_w += data.get('wins',0)
        nonedge_l += data.get('losses',0)

edge_total = edge_w + edge_l
nonedge_total = nonedge_w + nonedge_l

print()
print('─' * 60)
print('SUMMARY:')
if edge_total > 0:
    edge_wr = edge_w / edge_total * 100
    print(f'  Edge symbols:    {edge_w}W / {edge_l}L = {edge_wr:.1f}% ({edge_total} resolved)')
else:
    edge_wr = 0
    print(f'  Edge symbols:    No resolved trades')

if nonedge_total > 0:
    print(f'  Non-edge:        {nonedge_w}W / {nonedge_l}L = {nonedge_w/nonedge_total*100:.1f}% ({nonedge_total} resolved)')
else:
    print(f'  Non-edge:        No resolved trades (expected — whitelist should block these)')

print()
print('─' * 60)
print('VERDICT:')
if edge_total < 50:
    print(f'  ⚠️  INSUFFICIENT DATA: Only {edge_total} resolved edge trades.')
    print(f'     Need 50+ for statistical significance. Wait longer.')
elif edge_wr >= 60:
    print(f'  ✅ EDGE WR = {edge_wr:.1f}% — PROCEED to small real money test')
elif edge_wr >= 50:
    print(f'  ⚠️  EDGE WR = {edge_wr:.1f}% — REBUILD (Option B)')
else:
    print(f'  ❌ EDGE WR = {edge_wr:.1f}% — STOP (Option C)')
print('─' * 60)
"

echo ""
echo "Done."
