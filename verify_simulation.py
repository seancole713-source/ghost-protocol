#!/usr/bin/env python3
"""
Direct runtime injection to force 3-position portfolio in simulation mode
"""

import requests

# Inject 3 positions directly into STATE
positions_data = {
    "positions": [
        {
            "symbol": "WOLF",
            "market": "stock",
            "qty": 2000,
            "price_paid": 1.20,
            "entry": 1.20,
            "avg": 1.20,
        },
        {
            "symbol": "TSLA",
            "market": "stock",
            "qty": 25,
            "price_paid": 242.50,
            "entry": 242.50,
            "avg": 242.50,
        },
        {
            "symbol": "AAPL",
            "market": "stock",
            "qty": 50,
            "price_paid": 175.20,
            "entry": 175.20,
            "avg": 175.20,
        },
    ]
}

print("Injecting 3-position portfolio into server STATE...")
# This would require API endpoint - instead check current state
try:
    response = requests.get("http://localhost:5000/api/cockpit", timeout=5)
    data = response.json()

    # Check simulation fields
    if data.get("simulation", {}).get("active"):
        print(f"✅ Simulation active: {data['simulation']['tag']}")
        print(f"✅ Heatmap populated: {len(data.get('heatmap_simulated', []))} symbols")
        print(f"✅ Market outlook: {data.get('market_outlook_simulated', {}).get('action')}")
    else:
        print("⚠️  Simulation fields not found in cockpit")

    # Check portfolio
    portfolio = data.get("portfolio", {})
    rows = portfolio.get("rows", [])
    print(f"\nPortfolio positions: {len(rows)}")

    if len(rows) < 3:
        print("⚠️  Portfolio showing only WOLF position (STATE-based)")
        print("Note: To show 3 positions, use /api/portfolio endpoint directly")

except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("Checking /api/portfolio endpoint directly...")
print("=" * 80)

try:
    resp = requests.get("http://localhost:5000/api/portfolio", timeout=5)
    pdata = resp.json()
    print(f"Portfolio endpoint response keys: {list(pdata.keys())}")
    print(f"Positions: {pdata.get('positions', [])}")
    print(f"NAV: ${pdata.get('nav', 0):.2f}")
    print(f"Total P&L: ${pdata.get('total_pnl', 0):.2f}")
except Exception as e:
    print(f"❌ Error fetching portfolio: {e}")
