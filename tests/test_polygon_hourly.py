#!/usr/bin/env python3
"""Test Polygon hourly bars (more compatible with free tier)"""
import os
import requests
import time
from datetime import datetime, timedelta

api_key = os.getenv("POLYGON_API_KEY", "8VIvELVXiLG30K2l1348RzSurffLM0jR")

# Test: 48 hours ago (typical Ghost reconciliation window)
symbol = "AAPL"
timestamp = time.time() - (48 * 3600)
dt = datetime.fromtimestamp(timestamp)
target_date = dt.date()

# Add ±1 day buffer
start_date = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
end_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/hour/{start_date}/{end_date}"
params = {"apiKey": api_key, "sort": "asc", "limit": 1000}

print(f"Testing Polygon Hourly Bars...")
print(f"Symbol: {symbol}")
print(f"Target Time: {dt} (timestamp={timestamp})")
print(f"Date Range: {start_date} to {end_date}")
print(f"URL: {url}")
print()

try:
    response = requests.get(url, params=params, timeout=10)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Results Count: {data.get('resultsCount', 0)}")
        
        results = data.get('results', [])
        if results:
            print(f"\nFound {len(results)} hourly bars")
            
            # Find closest to target
            closest = min(results, key=lambda r: abs(r["t"]/1000 - timestamp))
            bar_time = closest["t"] / 1000
            bar_dt = datetime.fromtimestamp(bar_time)
            time_diff_hours = abs(bar_time - timestamp) / 3600
            
            print(f"\nClosest Bar:")
            print(f"  Time: {bar_dt}")
            print(f"  Price (close): ${closest['c']:.2f}")
            print(f"  Time Difference: {time_diff_hours:.1f} hours")
            print(f"  Open: ${closest['o']:.2f}")
            print(f"  High: ${closest['h']:.2f}")
            print(f"  Low: ${closest['l']:.2f}")
            print(f"  Volume: {closest['v']:,}")
            
            if time_diff_hours < 12:
                print(f"\n✅ ACCEPTABLE: Price within 12-hour tolerance")
            else:
                print(f"\n⚠️  WARNING: Price is {time_diff_hours:.1f}h away from target")
        else:
            print("\n❌ NO BARS FOUND")
            print(f"Response: {response.text[:500]}")
    else:
        print(f"❌ API Error: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()
