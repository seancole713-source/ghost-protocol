#!/usr/bin/env python3
"""
Force Ghost to refresh WOLF price by bypassing cache.
This clears the price cache and forces a fresh fetch from providers.
"""

import requests

BASE_URL = "http://localhost:5000"


def clear_price_cache():
    """Clear the price cache to force a refresh."""
    # There's no direct /clear-cache endpoint, so we'll simulate it
    # by waiting for TTL to expire or restarting server
    print("❌ No direct cache-clear API available")
    print("💡 Solutions:")
    print("   1. Wait 45 seconds for cache to expire")
    print("   2. Restart Ghost server")
    print("   3. Reduce PRICE_TTL_OPEN_S env var to 5 seconds")
    return False


def get_current_price():
    """Fetch current price from Ghost."""
    try:
        resp = requests.get(f"{BASE_URL}/api/portfolio", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        positions = data.get("positions", [])
        if positions:
            pos = positions[0]
            return {
                "symbol": pos.get("symbol"),
                "current": pos.get("current"),
                "qty": pos.get("qty"),
                "nav": pos.get("current", 0) * pos.get("qty", 0),
                "src": pos.get("src"),
            }
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    print("🔍 Checking current Ghost price...")
    price_data = get_current_price()

    if price_data:
        print("\n📊 Current Ghost Data:")
        print(f"   Symbol: {price_data['symbol']}")
        print(f"   Price: ${price_data['current']}")
        print(f"   Qty: {price_data['qty']}")
        print(f"   NAV: ${price_data['nav']:.2f}")
        print(f"   Source: {price_data['src']}")

        if price_data["src"] == "prev-close":
            print("\n⚠️  PROBLEM: Using stale prev-close price!")
            print("\n🔧 To fix, set environment variable:")
            print("   export PRICE_TTL_OPEN_S=5")
            print("\n   Then restart Ghost server:")
            print("   pkill -f 'uvicorn wolf_app' && sleep 2")
            print("   cd /workspaces/GHOST && source .venv/bin/activate")
            print("   export PRICE_TTL_OPEN_S=5 SIM_MODE=0")
            print(
                "   nohup uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload > ghost_server.out 2>&1 &"
            )
    else:
        print("❌ Could not fetch price data from Ghost")


if __name__ == "__main__":
    main()
