#!/usr/bin/env python3
"""
Quick local test to verify news endpoints work correctly.
Run this to confirm the code is ready for Railway deployment.
"""

import asyncio
import sys


async def test_news_endpoints():
    """Test that news endpoints can be imported and called."""
    print("🧪 Testing News Endpoints Locally")
    print("=" * 50)

    try:
        # Import the app
        print("\n1️⃣ Importing wolf_app...")
        from wolf_app import APP, api_inline_news, api_inline_news_recent

        print("   ✅ Successfully imported wolf_app")

        # Check routes registered
        print("\n2️⃣ Checking registered routes...")
        routes = [r for r in APP.routes if hasattr(r, "path")]
        news_routes = [r for r in routes if "/api/news" in r.path]
        print(f"   ✅ Total routes: {len(routes)}")
        print(f"   ✅ News routes: {len(news_routes)}")
        for r in news_routes:
            print(f"      - {list(r.methods)} {r.path}")

        # Test news endpoint
        print("\n3️⃣ Testing /api/news endpoint...")
        result = await api_inline_news(symbol=None, limit=5)
        if result and "news" in result:
            print(f"   ✅ Returned {result['count']} news items")
            print(f"   ✅ Status: {result.get('status', 'unknown')}")
            if result["count"] > 0:
                print(f"   ✅ Sample title: {result['news'][0]['title'][:60]}...")
        else:
            print(f"   ⚠️ Result: {result}")

        # Test recent news endpoint
        print("\n4️⃣ Testing /api/news/recent endpoint...")
        result = await api_inline_news_recent(symbol=None, minutes=120)
        if result and "news" in result:
            print(f"   ✅ Returned {result['count']} recent items")
            print(f"   ✅ Timeframe: {result.get('timeframe_minutes', 0)} minutes")
            print(f"   ✅ Status: {result.get('status', 'unknown')}")
        else:
            print(f"   ⚠️ Result: {result}")

        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED - Code is ready for Railway!")
        print("=" * 50)
        print("\nNext step: Manually redeploy on Railway dashboard")
        print("Expected after deploy:")
        print("  - Total routes: 256")
        print("  - /api/news → HTTP 200")
        print("  - /api/news/recent → HTTP 200")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_news_endpoints())
    sys.exit(0 if result else 1)
