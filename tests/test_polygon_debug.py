#!/usr/bin/env python3
"""Quick debug of Polygon API"""
import os
import requests
import time

api_key = os.getenv("POLYGON_API_KEY", "8VIvELVXiLG30K2l1348RzSurffLM0jR")

# Test 1: Simple Polygon API call
symbol = "AAPL"
timestamp = time.time() - (48 * 3600)  # 48 hours ago
start_ms = int((timestamp - 300) * 1000)
end_ms = int((timestamp + 300) * 1000)

url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start_ms}/{end_ms}"
params = {"apiKey": api_key, "sort": "asc", "limit": 10}

print(f"Testing Polygon API...")
print(f"URL: {url}")
print(f"Params: {params}")
print()

try:
    response = requests.get(url, params=params, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"\nResponse Body:")
    print(response.text[:1000])
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nParsed JSON:")
        print(f"  Status: {data.get('status')}")
        print(f"  Results count: {len(data.get('results', []))}")
        if data.get('results'):
            print(f"  First bar: {data['results'][0]}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
