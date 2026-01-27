#!/usr/bin/env python3
"""
Trigger the outcomes migration via admin API endpoint.
Runs inside Railway container where postgres.railway.internal resolves.
"""
import requests
import json

GHOST_URL = "https://ghost-protocol-production.up.railway.app"

print("=" * 60)
print("GHOST OUTCOMES MIGRATION TRIGGER")
print("=" * 60)
print(f"Target: {GHOST_URL}")
print()

try:
    print("📡 Calling migration endpoint...")
    response = requests.post(
        f"{GHOST_URL}/api/admin/migrate/outcomes",
        timeout=30
    )
    
    print(f"Status: {response.status_code}")
    print()
    
    result = response.json()
    print(json.dumps(result, indent=2))
    
    if result.get("ok"):
        print()
        print("=" * 60)
        print("✅ MIGRATION SUCCESSFUL")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Wait 1 hour for reconciler to run")
        print("2. Check outcomes: railway connect postgres")
        print("   SELECT COUNT(*) FROM ghost_prediction_outcomes;")
        print("3. Run audit when ready:")
        print("   railway run python3 analysis/accuracy_audit.py")
    else:
        print()
        print("=" * 60)
        print("❌ MIGRATION FAILED")
        print("=" * 60)
        print()
        print(f"Error: {result.get('error')}")
        if 'traceback' in result:
            print()
            print("Traceback:")
            print(result['traceback'])
    
except requests.exceptions.RequestException as e:
    print(f"❌ Connection failed: {e}")
    print()
    print("Make sure Ghost is running on Railway:")
    print("  railway status")

except Exception as e:
    print(f"❌ Error: {e}")

print()
