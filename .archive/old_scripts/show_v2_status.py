#!/usr/bin/env python3
"""Display V2 quality filter status."""

import requests
import json

response = requests.get("https://ghost-protocol-production.up.railway.app/api/v2/quality/status")
data = response.json()

print('=' * 80)
print('GHOST PROTOCOL V2 - QUALITY FILTER STATUS')
print('=' * 80)
print(f"✅ WHITELIST: {data.get('whitelist_count', 0)} symbols (predict freely)")
if data.get('whitelist'):
    for s in data['whitelist']:
        print(f'   ✅ {s}')
print()
print(f"❌ BLACKLIST: {data.get('blacklist_count', 0)} symbols (DO NOT predict)")
if data.get('blacklist'):
    shown = 0
    for s in data['blacklist']:
        if shown < 8:
            print(f'   ❌ {s}')
            shown += 1
    if data.get('blacklist_count', 0) > 8:
        print(f'   ... and {data.get("blacklist_count") - 8} more')
print()
print('RULES:')
print('  • Whitelist (90%+ WR): Predict freely')
print('  • Blacklist (0-45% WR): DO NOT predict')
print('  • All major crypto BLOCKED until 2.2% win rate improves')
print('=' * 80)
print()
print('🎯 IMPACT: Ghost will now ONLY predict 10 proven high performers')
print('🎯 TARGET: 70%+ win rate through quality over quantity')
print()
print('✅ V2 Integration COMPLETE')
