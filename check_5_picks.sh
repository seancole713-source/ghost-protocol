#!/bin/bash
# Run this when market opens (8:30 AM CT on Jan 2)
# ./check_5_picks.sh

echo "=== 5 EXPIRING PICKS - LIVE PRICES ==="
echo "Checking at: $(date)"
echo ""

curl -s "https://ghost-protocol-production.up.railway.app/debug/tracking-status?secret=o4Cf0tKYUkjPkjXddbtLcLBZeZ7YnxYc" | python3 -c "
import json, sys, urllib.request

d = json.load(sys.stdin)
picks = d.get('active_picks', [])

# The 5 that expire Jan 2
expiring = ['GOOGL', 'TSLA', 'ADBE', 'CSCO', 'AAPL']

print(f\"{'SYMBOL':<8} {'DIR':<5} {'ENTRY':>10} {'TARGET':>10} {'STOP':>10} {'CURRENT':>10} {'P&L':>8} {'STATUS'}\")
print('='*85)

for p in picks:
    sym = p['symbol']
    if sym not in expiring:
        continue
    
    entry = p['entry_price']
    target = p['target_price']
    stop = p['stop_price']
    direction = p['direction']
    
    # Get live price
    try:
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=1d'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.load(resp)
        current = data['chart']['result'][0]['meta']['regularMarketPrice']
    except:
        current = entry
    
    # Calculate P&L
    if direction == 'BUY':
        pnl = (current - entry) / entry * 100
        hit_target = current >= target
        hit_stop = current <= stop
    else:
        pnl = (entry - current) / entry * 100
        hit_target = current <= target
        hit_stop = current >= stop
    
    # Status
    if hit_target:
        status = '�� TARGET HIT!'
    elif hit_stop:
        status = '💀 STOP HIT!'
    elif pnl > 0:
        status = '🟢 Winning'
    elif pnl < -1:
        status = '🔴 Losing'
    else:
        status = '⚪ Flat'
    
    print(f'{sym:<8} {direction:<5} \${entry:>9.2f} \${target:>9.2f} \${stop:>9.2f} \${current:>9.2f} {pnl:>+7.2f}% {status}')

print()
print('These picks expire TODAY at 9:16 AM CT')
"
