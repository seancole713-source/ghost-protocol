#!/usr/bin/env python3
import requests
import time

BASE = "https://ghost-protocol-production.up.railway.app"

print("Testing server responsiveness...")
print("-" * 60)

# Test 1: Simple health check with increasing timeouts
for timeout_s in [2, 5, 10, 20]:
    try:
        start = time.time()
        r = requests.get(f"{BASE}/health", timeout=timeout_s)
        elapsed = time.time() - start
        
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Health OK ({elapsed:.1f}s, timeout={timeout_s}s)")
            print(f"   Uptime: {data.get('uptime', 0)}s")
            break
    except requests.exceptions.Timeout:
        print(f"❌ Timeout at {timeout_s}s")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}")
        
print()

# Test 2: Hunter feed
print("Testing hunter feed...")
try:
    r = requests.get(f"{BASE}/api/v3/hunter/feed?limit=1", timeout=15)
    data = r.json()
    print(f"✅ Hunter feed: {data.get('count', 0)} predictions")
except requests.exceptions.Timeout:
    print(f"❌ Hunter feed: Timeout")
except Exception as e:
    print(f"❌ Hunter feed: {type(e).__name__}: {e}")

print()
print("If all tests timeout, the server is likely overloaded.")
print("Check Railway logs for memory/CPU issues or long-running processes.")
