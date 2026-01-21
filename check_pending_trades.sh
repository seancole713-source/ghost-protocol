#!/bin/bash
# Check pending trades via API endpoint

echo "🔍 Checking paper trade stats from Ghost Protocol API..."
echo ""

curl -s "https://ghost-protocol-production.up.railway.app/api/v3/paper/stats" | python3 -c "
import sys, json
from datetime import datetime

data = json.load(sys.stdin)
stats = data.get('stats', {})

total = stats.get('total_trades', 0)
pending = stats.get('pending_trades', 0)
resolved = stats.get('resolved_trades', 0)
wins = stats.get('wins', 0)
losses = stats.get('losses', 0)
win_rate = stats.get('win_rate', 0) * 100

print(f'📊 OVERALL PAPER TRADING STATS')
print(f'=' * 50)
print(f'Total Trades:    {total:,}')
print(f'Pending Trades:  {pending:,} ⚠️')
print(f'Resolved Trades: {resolved:,}')
print(f'Wins:            {wins:,}')
print(f'Losses:          {losses:,}')
print(f'Win Rate:        {win_rate:.1f}%')
print()

# Show V2 whitelisted symbols
v2_whitelist = ['CHZ', 'ZEC', 'RNDR', 'ILV', 'T', 'TURBO', 'RLC', 'EGLD', 'LRC', 'ICP', 'OCEAN']
print(f'🎯 V2 WHITELISTED SYMBOLS PERFORMANCE')
print(f'=' * 50)

accuracy_by_symbol = stats.get('accuracy_by_symbol', {})

total_v2_wins = 0
total_v2_trades = 0

for symbol in sorted(v2_whitelist):
    if symbol in accuracy_by_symbol:
        sym_stats = accuracy_by_symbol[symbol]
        trades = sym_stats.get('trades', 0)
        wins = sym_stats.get('wins', 0)
        wr = sym_stats.get('win_rate', 0) * 100
        
        if trades > 0:
            total_v2_wins += wins
            total_v2_trades += trades
            status = '✅' if wr >= 80 else '⚠️' if wr >= 50 else '❌'
            print(f'{symbol:8} {trades:3} trades | {wins:3} wins | {wr:5.1f}% {status}')

if total_v2_trades > 0:
    v2_win_rate = (total_v2_wins / total_v2_trades) * 100
    print()
    print(f'V2 Combined:  {total_v2_trades:,} trades | {total_v2_wins:,} wins | {v2_win_rate:.1f}% win rate')

print()
print(f'🚨 ISSUE: {pending:,} pending trades are bloating the database')
print(f'These are likely pre-V2 filter trades from before Jan 14, 2026')
"

echo ""
echo "💡 RECOMMENDATION:"
echo "Create an endpoint to expire old pending trades and improve stats accuracy"
