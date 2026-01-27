#!/usr/bin/env python3
"""
Ghost Protocol Production Verification
Tests all endpoints on the live Railway deployment
"""

import requests

BASE_URL = "https://web-production-8e9a0.up.railway.app"


def test_endpoint(name, path, expected_keys=None):
    """Test a single endpoint"""
    url = f"{BASE_URL}{path}"
    try:
        response = requests.get(url, timeout=10)
        status = response.status_code

        if status == 200:
            data = response.json()
            if expected_keys:
                has_keys = all(key in data for key in expected_keys)
                if has_keys:
                    print(f"✅ {name}: {path}")
                    print(f"   Status: HTTP {status}")
                    print(f"   Keys: {list(data.keys())[:5]}")
                else:
                    print(f"⚠️  {name}: {path}")
                    print(f"   Status: HTTP {status} but missing expected keys")
                    print(f"   Expected: {expected_keys}")
                    print(f"   Got: {list(data.keys())}")
            else:
                print(f"✅ {name}: {path}")
                print(f"   Status: HTTP {status}")
                print(f"   Keys: {list(data.keys())[:5]}")
        elif status == 404:
            print(f"❌ {name}: {path}")
            print("   Status: HTTP 404 - Endpoint not found")
        elif status == 422:
            print(f"⚠️  {name}: {path}")
            print("   Status: HTTP 422 - Missing required parameters")
        elif status == 500:
            print(f"❌ {name}: {path}")
            print("   Status: HTTP 500 - Internal Server Error")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            except Exception:
                pass
        else:
            print(f"⚠️  {name}: {path}")
            print(f"   Status: HTTP {status}")
        print()
        return status == 200
    except requests.exceptions.Timeout:
        print(f"❌ {name}: {path}")
        print("   Error: Request timeout (10s)")
        print()
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: {path}")
        print("   Error: Connection failed - server may be down")
        print()
        return False
    except Exception as e:
        print(f"❌ {name}: {path}")
        print(f"   Error: {str(e)}")
        print()
        return False


def main():
    print("=" * 80)
    print("🚀 GHOST PROTOCOL PRODUCTION VERIFICATION")
    print(f"🌐 Testing: {BASE_URL}")
    print("=" * 80)
    print()

    # Test health check first
    print("🏥 HEALTH CHECK")
    print("-" * 80)
    health_ok = test_endpoint("Health Check", "/health", ["ok", "ts"])

    if not health_ok:
        print("❌ Server is not responding! Check Railway deployment logs.")
        print("   Visit: https://railway.app")
        return

    print()
    print("🤖 AGENT ENDPOINTS")
    print("-" * 80)
    test_endpoint("Agent Decisions", "/api/agent/decisions", ["decisions", "count"])
    test_endpoint("Agent Stats", "/api/agent/stats", ["total_decisions", "timestamp"])

    print()
    print("📰 NEWS ENDPOINTS")
    print("-" * 80)
    test_endpoint("News Feed", "/api/news", ["news", "count"])
    test_endpoint("News Recent", "/api/news/recent", ["news", "count"])

    print()
    print("📊 SNAPSHOT ENDPOINTS")
    print("-" * 80)
    test_endpoint("System Snapshot", "/api/snapshot", ["timestamp", "portfolio"])
    test_endpoint("Research Snapshot WOLF", "/api/research/snapshot/WOLF", ["symbol", "timestamp"])

    print()
    print("📈 STAGE 2 - FORECASTING")
    print("-" * 80)
    test_endpoint("Stage2 Forecasts", "/api/stage2/forecasts", ["forecasts", "count"])
    test_endpoint("Stage2 Accuracy", "/api/stage2/accuracy")

    print()
    print("💼 PORTFOLIO ENDPOINTS")
    print("-" * 80)
    test_endpoint("Portfolio", "/api/portfolio", ["positions", "cash", "nav"])

    print()
    print("🎯 STAGE 3 - RISK & REGIME")
    print("-" * 80)
    test_endpoint("Market Regime", "/api/stage3/regime/current", ["regime", "confidence"])
    test_endpoint("Risk Dashboard", "/api/stage3/risk/dashboard", ["portfolio", "status"])

    print()
    print("⚡ STAGE 5 - EXECUTION")
    print("-" * 80)
    test_endpoint("Execution Analytics", "/api/stage5/execution/analytics", ["timestamp"])

    print()
    print("📊 PRICE & MARKET DATA")
    print("-" * 80)
    test_endpoint("Market Mood", "/api/stage1/mood")
    test_endpoint("World Context", "/api/stage1/world")

    print()
    print("=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)
    print()
    print("🌐 Access Ghost Cockpit:")
    print(f"   {BASE_URL}/cockpit")
    print()
    print("📊 Check Railway logs:")
    print("   https://railway.app")
    print()


if __name__ == "__main__":
    main()
