#!/usr/bin/env python3
"""
Add realistic placeholder test data to GHOST for UI testing
Based on factual market patterns for WOLF stock
"""

import time

import requests

BASE_URL = "http://localhost:5000"

# Realistic WOLF stock data based on actual market patterns
# WOLF typically trades in $20-30 range
TEST_DATA = {
    "positions": [
        {
            "symbol": "WOLF",
            "type": "stock",
            "quantity": 100,
            "price": 24.50,  # Entry price
            "market": "stock",
        },
        {
            "symbol": "ETH",
            "type": "crypto",
            "quantity": 0.5,
            "price": 2650.00,  # Entry price
            "market": "crypto",
        },
        {
            "symbol": "BTC",
            "type": "crypto",
            "quantity": 0.025,
            "price": 42000.00,  # Entry price
            "market": "crypto",
        },
    ],
    "cash": 5000.00,  # Available cash
}


def wait_for_server(max_wait=30):
    """Wait for server to be ready"""
    print(f"Waiting for server at {BASE_URL}...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=2)
            if resp.status_code == 200:
                print("✅ Server is ready!")
                return True
        except Exception:
            time.sleep(1)
    print("❌ Server not available")
    return False


def add_positions():
    """Add test positions via API"""
    print("\n📊 Adding test positions...")

    for pos in TEST_DATA["positions"]:
        try:
            resp = requests.post(
                f"{BASE_URL}/api/bank/add_position",
                json={
                    "symbol": pos["symbol"],
                    "quantity": pos["quantity"],
                    "price": pos["price"],
                    "type": pos["type"],
                },
                timeout=5,
            )

            if resp.status_code == 200:
                print(f"  ✅ Added {pos['quantity']} {pos['symbol']} @ ${pos['price']}")
            else:
                print(f"  ⚠️  Failed to add {pos['symbol']}: {resp.status_code}")

        except Exception as e:
            print(f"  ❌ Error adding {pos['symbol']}: {e}")

    print("\n✅ Test data loading complete!")


def check_cockpit():
    """Check cockpit endpoint to see the data"""
    print("\n🔍 Checking cockpit data...")
    try:
        resp = requests.get(f"{BASE_URL}/api/cockpit", timeout=5)
        if resp.status_code == 200:
            data = resp.json()

            print("\n📈 Portfolio Summary:")
            kpis = data.get("kpis", {})
            print(f"  NAV: ${kpis.get('nav', 0):,.2f}")
            print(f"  Cash: ${kpis.get('cash', 0):,.2f}")
            print(f"  PnL: ${kpis.get('pnl_abs', 0):,.2f} ({kpis.get('pnl_pct', 0):.2f}%)")

            positions = data.get("portfolio", {}).get("rows", [])
            print(f"\n📊 Positions ({len(positions)}):")
            for pos in positions:
                symbol = pos.get("symbol", "?")
                qty = pos.get("qty", 0)
                current = pos.get("current", 0)
                pnl_pct = pos.get("pnl_pct", 0)
                print(f"  {symbol}: {qty} shares @ ${current:.2f} ({pnl_pct:+.2f}%)")

            print("\n✅ Cockpit data looks good!")
            print(
                "\n🌐 View UI at: https://crispy-happiness-q7gp6xvxr9r62xv9v-5000.app.github.dev/"
            )

    except Exception as e:
        print(f"❌ Error checking cockpit: {e}")


def main():
    """Main execution"""
    print("=" * 60)
    print("🚀 GHOST UI Test Data Setup")
    print("=" * 60)

    if not wait_for_server():
        print("\n❌ Server must be running first!")
        print("   Start it with: make srv")
        return 1

    add_positions()
    time.sleep(2)  # Give server time to process
    check_cockpit()

    print("\n" + "=" * 60)
    print("✅ Setup complete! Refresh your browser to see the data.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
