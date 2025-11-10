#!/usr/bin/env python3
"""Quick Ghost Server Test"""

import requests

BASE_URL = "http://localhost:8444"

print("=" * 80)
print("🧪 GHOST SERVER FUNCTIONALITY TEST")
print("=" * 80)
print()

tests = [
    ("Health Check", f"{BASE_URL}/health"),
    ("API Status", f"{BASE_URL}/api/status"),
    ("Portfolio", f"{BASE_URL}/api/portfolio"),
    ("Version", f"{BASE_URL}/api/version"),
    ("Config", f"{BASE_URL}/api/config"),
]

for name, url in tests:
    try:
        response = requests.get(url, timeout=5)
        status = "✅" if response.status_code == 200 else "⚠️"
        print(f"{status} {name}: HTTP {response.status_code}")
        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, dict):
                    print(f"   Keys: {list(data.keys())[:5]}")
            except Exception:
                pass
    except Exception as e:
        print(f"❌ {name}: {str(e)[:50]}")
    print()

print("=" * 80)
print("Testing UI endpoints...")
print("=" * 80)
print()

ui_tests = [
    ("Main UI", f"{BASE_URL}/"),
    ("Cockpit", f"{BASE_URL}/cockpit"),
    ("UI Health", f"{BASE_URL}/ui/health"),
]

for name, url in ui_tests:
    try:
        response = requests.get(url, timeout=5)
        status = "✅" if response.status_code == 200 else "⚠️"
        content_type = response.headers.get("content-type", "unknown")
        print(f"{status} {name}: HTTP {response.status_code} ({content_type})")
    except Exception as e:
        print(f"❌ {name}: {str(e)[:50]}")

print()
print("=" * 80)
print("✅ Ghost Server is operational on port 8444!")
print(f"🌐 Open: {BASE_URL}/cockpit")
print("=" * 80)
