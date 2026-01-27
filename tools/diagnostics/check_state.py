#!/usr/bin/env python3
"""Quick STATE diagnostic"""

import sys

sys.path.insert(0, "/workspaces/GHOST")

# Import after path is set
from wolf_app import STATE, WOLF

print("🔍 GHOST STATE DIAGNOSTIC")
print("=" * 60)
print(f"\nWOLF Symbol: {WOLF}")
print(f"STATE['qty']: {STATE.get('qty', 'NOT SET')}")
print(f"STATE['avg_cost']: {STATE.get('avg_cost', 'NOT SET')}")
print(f"STATE['cash']: {STATE.get('cash', 'NOT SET')}")
print(f"STATE['mode']: {STATE.get('mode', 'NOT SET')}")
print(f"\nSTATE['positions']: {STATE.get('positions', 'NOT SET')}")
print(f"\nFull STATE keys: {list(STATE.keys())}")
print("=" * 60)
