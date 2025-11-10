#!/usr/bin/env python3
"""
Check what routes are actually registered in the deployed app
"""

import requests

BASE_URL = "https://web-production-8e9a0.up.railway.app"

# Try to get all routes from FastAPI's OpenAPI schema
try:
    response = requests.get(f"{BASE_URL}/openapi.json", timeout=10)
    if response.status_code == 200:
        schema = response.json()
        print("🔍 ALL REGISTERED ROUTES:")
        print("=" * 80)

        paths = schema.get("paths", {})
        for path in sorted(paths.keys()):
            methods = list(paths[path].keys())
            print(f"  {' '.join(m.upper() for m in methods):10} {path}")

        print()
        print(f"📊 Total routes: {len(paths)}")
        print()

        # Check specifically for our new routes
        news_routes = [p for p in paths.keys() if "news" in p.lower()]
        agent_routes = [p for p in paths.keys() if "agent" in p.lower()]
        snapshot_routes = [p for p in paths.keys() if "snapshot" in p.lower()]

        print("🎯 NEW ROUTES STATUS:")
        print("-" * 80)
        print(f"News routes found: {len(news_routes)}")
        for route in news_routes:
            print(f"  ✅ {route}")

        print(f"\nAgent routes found: {len(agent_routes)}")
        for route in agent_routes:
            print(f"  ✅ {route}")

        print(f"\nSnapshot routes found: {len(snapshot_routes)}")
        for route in snapshot_routes:
            print(f"  ✅ {route}")

    else:
        print(f"❌ Could not get OpenAPI schema: HTTP {response.status_code}")
except Exception as e:
    print(f"❌ Error checking routes: {e}")
