#!/usr/bin/env python3
"""Query production database for 30-day win rates by symbol."""

import requests
import json

# Fetch data from V2 API
response = requests.get("https://ghost-protocol-production.up.railway.app/api/v2/performance/dashboard?days=30")
data = response.json()

if not data.get('ok'):
    print('ERROR: API returned error')
    exit(1)

# Extract symbol details
top_details = data.get('top_10_details', [])
bottom_details = data.get('bottom_10_details', [])

# Combine and sort all by win_rate
all_symbols = top_details + bottom_details
all_symbols.sort(key=lambda x: x['win_rate'], reverse=True)

print('=' * 80)
print('30-DAY WIN RATE BY SYMBOL (Production Database)')
print('=' * 80)
print(f"{'SYMBOL':<12} {'TYPE':<10} {'TOTAL':>8} {'WINS':>8} {'WIN RATE':>12} {'TREND':<12}")
print('-' * 80)

for symbol in all_symbols:
    name = symbol['symbol']
    asset_type = symbol['asset_type']
    total = symbol['total']
    wins = symbol['wins']
    wr = symbol['win_rate']
    trend = symbol.get('trend', 'unknown')
    print(f"{name:<12} {asset_type:<10} {total:>8} {wins:>8} {wr:>11.1f}% {trend:<12}")

print('=' * 80)
print(f"Total unique symbols: {len(all_symbols)}")
print()

# Overall stats
overall = data.get('overall', {})
print(f"OVERALL (30 days): {overall.get('wins', 0)}/{overall.get('total_predictions', 0)} wins = {overall.get('win_rate', 0):.1f}%")
print()
print("By asset type:")
by_type = data.get('by_asset_type', {})
for asset_type, stats in by_type.items():
    print(f"  {asset_type.upper()}: {stats.get('wins', 0)}/{stats.get('total', 0)} = {stats.get('win_rate', 0):.1f}%")
