#!/usr/bin/env python3
"""Test UI Data Endpoints"""

import requests

BASE_URL = "http://localhost:8444"

print("=" * 80)
print("🔍 TESTING UI DATA ENDPOINTS")
print("=" * 80)
print()

# Test endpoints that UI panels are trying to load
endpoints = [
    ("Agent Decisions", "/api/agent/decisions"),
    ("Agent Stats", "/api/agent/stats"),
    ("World Context", "/api/stage1/world"),
    ("Market Mood", "/api/stage1/mood"),
    ("Forecasts", "/api/predict/history"),
    ("Forecast Series", "/api/predict/series"),
    ("Stage2 Forecasts", "/api/stage2/forecasts"),
    ("Stage2 Accuracy", "/api/stage2/accuracy"),
    ("News Feed", "/api/news"),
    ("News Recent", "/api/news/recent"),
    ("Research Snapshot", "/api/research/snapshot/WOLF"),
    ("Portfolio", "/api/portfolio"),
    ("Regime", "/api/stage3/regime/current"),
    ("Risk Dashboard", "/api/stage3/risk/dashboard"),
    ("Stage4 Portfolio", "/api/stage4/portfolio/optimize"),
    ("Execution Analytics", "/api/stage5/execution/analytics"),
]

for name, endpoint in endpoints:
    try:
        response = requests.get(BASE_URL + endpoint, timeout=5)
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ {name}: {endpoint}")
                print(f"   Status: HTTP {response.status_code}")
                if isinstance(data, dict):
                    print(f"   Keys: {list(data.keys())[:8]}")
                elif isinstance(data, list):
                    print(f"   Array length: {len(data)}")
                else:
                    print(f"   Type: {type(data)}")
            except Exception:
                print(f"✅ {name}: {endpoint}")
                print(f"   Status: HTTP {response.status_code} (non-JSON)")
        elif response.status_code == 404:
            print(f"❌ {name}: {endpoint}")
            print("   Status: HTTP 404 - Endpoint not found")
        elif response.status_code == 422:
            print(f"⚠️  {name}: {endpoint}")
            print("   Status: HTTP 422 - Missing required parameters")
        elif response.status_code == 405:
            print(f"⚠️  {name}: {endpoint}")
            print("   Status: HTTP 405 - Wrong method (might need POST)")
        else:
            print(f"⚠️  {name}: {endpoint}")
            print(f"   Status: HTTP {response.status_code}")
            print(f"   Error: {response.text[:100]}")
    except requests.exceptions.Timeout:
        print(f"⏱️  {name}: {endpoint}")
        print("   Error: Request timeout")
    except Exception as e:
        print(f"❌ {name}: {endpoint}")
        print(f"   Error: {str(e)[:80]}")
    print()

print("=" * 80)
print("Testing specific data that UI is looking for...")
print("=" * 80)
print()

# Test snapshot endpoint that UI is polling
try:
    response = requests.get(f"{BASE_URL}/api/snapshot", timeout=5)
    print(f"Snapshot endpoint: HTTP {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Snapshot keys: {list(data.keys())}")
        print(f"  Price: {data.get('price', 'N/A')}")
        print(f"  GPS: {data.get('gps', 'N/A')}")
except Exception as e:
    print(f"Snapshot error: {e}")

print()
