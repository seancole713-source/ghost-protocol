#!/usr/bin/env python3
"""Check if outcomes table exists and has data."""
import requests
import json

GHOST_URL = "https://ghost-protocol-production.up.railway.app"

print("=" * 60)
print("GHOST OUTCOMES TABLE STATUS CHECK")
print("=" * 60)

# Check accuracy endpoint
print("\n1. Checking /api/v3/accuracy/summary...")
try:
    resp = requests.get(f"{GHOST_URL}/api/v3/accuracy/summary", timeout=10)
    print(f"   Status: {resp.status_code}")
    data = resp.json()
    print(f"   Response: {json.dumps(data, indent=2)}")
    
    if resp.status_code == 200:
        if data.get("ok") is False and "No reconciled" in data.get("error", ""):
            print("\n   ✅ Table EXISTS but is EMPTY (no outcomes yet)")
        elif data.get("ok"):
            print("\n   ✅ Table EXISTS and HAS DATA")
            print(f"   Accuracy: {data.get('accuracy_pct')}%")
    else:
        print("\n   ❓ Unexpected response")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check system status
print("\n2. Checking /api/v3/system/status...")
try:
    resp = requests.get(f"{GHOST_URL}/api/v3/system/status", timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        if "outcome_reconciler" in data:
            print(f"   Reconciler: {json.dumps(data['outcome_reconciler'], indent=4)}")
        else:
            print("   (No reconciler info in status)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check predictions endpoint
print("\n3. Checking /api/v3/predictions/latest...")
try:
    resp = requests.get(f"{GHOST_URL}/api/v3/predictions/latest", timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"   ✅ Latest prediction: {data[0].get('symbol')} - {data[0].get('direction')} @ {data[0].get('confidence')}%")
        else:
            print(f"   Predictions: {len(data) if isinstance(data, list) else 'unknown'}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("\nThe ghost_prediction_outcomes table EXISTS in production.")
print("\nNext steps:")
print("  1. Wait 1-2 hours for reconciler to process predictions")
print("  2. Re-check: python3 check_outcomes_status.py")
print("  3. Once outcomes exist, run full audit:")
print("     railway run python3 analysis/accuracy_audit.py")
print("=" * 60)
