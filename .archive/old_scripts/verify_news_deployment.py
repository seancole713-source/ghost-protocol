#!/usr/bin/env python3
"""
Verify news router deployment on Railway.
Tests all news endpoints and debug endpoint.
"""

import requests

BASE_URL = "https://web-production-8e9a0.up.railway.app"


def test_endpoint(name, path, expected_status=200):
    """Test a single endpoint."""
    url = f"{BASE_URL}{path}"
    try:
        response = requests.get(url, timeout=10)
        status = "✅" if response.status_code == expected_status else "❌"
        print(f"{status} {name}: HTTP {response.status_code}")

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"   Keys: {list(data.keys())}")
                return True, data
            except Exception:
                print(f"   Response: {response.text[:100]}")
                return False, None
        else:
            print(f"   Error: {response.text[:100]}")
            return False, None
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False, None


def main():
    print("=" * 80)
    print("🚀 GHOST NEWS ROUTER DEPLOYMENT VERIFICATION")
    print(f"🌐 Testing: {BASE_URL}")
    print("=" * 80)

    results = {}

    # 1. Test debug endpoint first
    print("\n🔍 DEBUG ENDPOINT")
    print("-" * 80)
    success, data = test_endpoint("Router Status", "/debug/router_status")
    if success and data:
        results["debug"] = data
        print(f"   news_router_mounted: {data.get('news_router_mounted')}")
        print(f"   routes_dir_exists: {data.get('routes_dir_exists')}")
        print(f"   total_routes: {data.get('total_routes')}")
        print(f"   news_routes: {data.get('news_routes')}")

    # 2. Test news endpoints
    print("\n📰 NEWS ENDPOINTS")
    print("-" * 80)

    success, data = test_endpoint("News Feed", "/api/news?limit=5")
    if success and data:
        results["news"] = data
        print(f"   Articles: {data.get('count', 0)}")

    success, data = test_endpoint("News Recent", "/api/news/recent?minutes=120")
    if success and data:
        results["news_recent"] = data
        print(f"   Articles: {data.get('count', 0)}")

    success, data = test_endpoint("News Sentiment", "/api/news/sentiment/WOLF")
    if success and data:
        results["sentiment"] = data
        print(f"   Symbol: {data.get('symbol')}")

    # 3. Test existing endpoints still work
    print("\n✅ EXISTING ENDPOINTS (Should still work)")
    print("-" * 80)
    test_endpoint("Health", "/health")
    test_endpoint("Agent Decisions", "/api/agent/decisions")
    test_endpoint("Snapshot", "/api/snapshot")

    # 4. Check OpenAPI schema
    print("\n📋 OPENAPI SCHEMA")
    print("-" * 80)
    try:
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=10)
        if response.status_code == 200:
            schema = response.json()
            all_paths = list(schema.get("paths", {}).keys())
            news_paths = [p for p in all_paths if "/news" in p.lower()]

            print(f"✅ Total routes in schema: {len(all_paths)}")
            print(f"📰 News routes in schema: {len(news_paths)}")
            for path in news_paths:
                print(f"   - {path}")
        else:
            print(f"❌ Failed to fetch OpenAPI schema: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Error fetching schema: {e}")

    # 5. Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    if "debug" in results:
        mounted = results["debug"].get("news_router_mounted", False)
        if mounted:
            print("✅ News router successfully mounted!")
            print("✅ All news endpoints should be working")
        else:
            error = results["debug"].get("news_router_error")
            print("❌ News router NOT mounted")
            print(f"   Error: {error}")
    else:
        print("❌ Could not verify router status (debug endpoint failed)")

    print("\n🌐 Access Ghost Cockpit:")
    print(f"   {BASE_URL}/cockpit")
    print("=" * 80)


if __name__ == "__main__":
    main()
