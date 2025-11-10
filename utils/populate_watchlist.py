#!/usr/bin/env python3
"""
Script to populate watchlist with initial market data and calculate GHOST scores.
Run this to set up the watchlist with your symbols.
"""

import sys

sys.path.insert(0, "/workspaces/GHOST")

from core.watchlist_manager import get_watchlist_manager

# Market data from user (current prices and stats)
INITIAL_MARKET_DATA = [
    {
        "symbol": "WFC",
        "name": "Wells Fargo & Company",
        "price": 80.67,
        "change_pct": 0.21,
        "volume": 8009000.0,
        "market_cap": 258422000000.00,
    },
    {
        "symbol": "SLB",
        "name": "Schlumberger Limited",
        "price": 34.26,
        "change_pct": 0.44,
        "volume": 9581000.0,
        "market_cap": 51180000000.0,
    },
    {
        "symbol": "HLN",
        "name": "Haleon plc",
        "price": 8.95,
        "change_pct": 0.79,
        "volume": 7887000.0,
        "market_cap": 39890000000.0,
    },
    {
        "symbol": "CNH",
        "name": "CNH Industrial N.V.",
        "price": 10.91,
        "change_pct": 0.65,
        "volume": 7730000.0,
        "market_cap": 13646000000.0,
    },
    {
        "symbol": "KDP",
        "name": "Keurig Dr Pepper Inc.",
        "price": 25.84,
        "change_pct": 0.39,
        "volume": 10064000.0,
        "market_cap": 35102000000.0,
    },
    {
        "symbol": "CORZ",
        "name": "Core Scientific, Inc.",
        "price": 17.82,
        "change_pct": -1.55,
        "volume": 8070000.0,
        "market_cap": 5477000000.0,
    },
    {
        "symbol": "SBUX",
        "name": "Starbucks Corporation",
        "price": 86.42,
        "change_pct": -0.35,
        "volume": 7628000.0,
        "market_cap": 98234000000.0,
    },
    {
        "symbol": "UWMC",
        "name": "UWM Holdings Corporation",
        "price": 5.98,
        "change_pct": -1.48,
        "volume": 7590000.0,
        "market_cap": 9639000000.0,
    },
    {
        "symbol": "EQT",
        "name": "EQT Corporation",
        "price": 56.03,
        "change_pct": 0.48,
        "volume": 7589000.0,
        "market_cap": 34966000000.0,
    },
    {
        "symbol": "MDT",
        "name": "Medtronic plc",
        "price": 97.7,
        "change_pct": 2.33,
        "volume": 7613000.0,
        "market_cap": 125318000000.0,
    },
    {
        "symbol": "HPQ",
        "name": "HP Inc.",
        "price": 26.64,
        "change_pct": 0.6,
        "volume": 7680000.0,
        "market_cap": 24900000000.0,
    },
    {
        "symbol": "ETSY",
        "name": "Etsy, Inc.",
        "price": 72.38,
        "change_pct": -0.22,
        "volume": 7473000.0,
        "market_cap": 7173000000.0,
    },
    {
        "symbol": "PBA",
        "name": "Pembina Pipeline Corporation",
        "price": 42.09,
        "change_pct": 6.02,
        "volume": 7355000.0,
        "market_cap": 24489000000.0,
    },
    {
        "symbol": "LVS",
        "name": "Las Vegas Sands Corp.",
        "price": 50.97,
        "change_pct": -7.41,
        "volume": 7416000.0,
        "market_cap": 34989000000.0,
    },
    {
        "symbol": "PGY",
        "name": "Pagaya Technologies Ltd.",
        "price": 29.4,
        "change_pct": -5.37,
        "volume": 7370000.0,
        "market_cap": 2252000000.0,
    },
    {
        "symbol": "CTRA",
        "name": "Coterra Energy Inc.",
        "price": 23.3,
        "change_pct": 0.65,
        "volume": 7246000.0,
        "market_cap": 17781000000.0,
    },
    {
        "symbol": "HBM",
        "name": "Hudbay Minerals Inc.",
        "price": 15.74,
        "change_pct": 1.88,
        "volume": 7244000.0,
        "market_cap": 6237000000.0,
    },
    {
        "symbol": "MRNA",
        "name": "Moderna, Inc.",
        "price": 28.49,
        "change_pct": 0.42,
        "volume": 8040000.0,
        "market_cap": 11085000000.0,
    },
    {
        "symbol": "SBSW",
        "name": "Sibanye Stillwater Limited",
        "price": 11.17,
        "change_pct": -1.76,
        "volume": 7147000.0,
        "market_cap": 7904000000.0,
    },
    {
        "symbol": "CVS",
        "name": "CVS Health Corporation",
        "price": 77.49,
        "change_pct": 0.05,
        "volume": 7083000.0,
        "market_cap": 98283000000.0,
    },
    {
        "symbol": "KHC",
        "name": "The Kraft Heinz Company",
        "price": 26.06,
        "change_pct": -0.04,
        "volume": 8579000.0,
        "market_cap": 30845000000.0,
    },
    {
        "symbol": "M",
        "name": "Macy's, Inc.",
        "price": 18.21,
        "change_pct": 0.11,
        "volume": 7024000.0,
        "market_cap": 4889000000.0,
    },
    {
        "symbol": "VTRS",
        "name": "Viatris Inc.",
        "price": 10.19,
        "change_pct": 0.79,
        "volume": 7008000.0,
        "market_cap": 11880000000.0,
    },
    {
        "symbol": "PDD",
        "name": "PDD Holdings Inc.",
        "price": 134.25,
        "change_pct": -0.73,
        "volume": 6999000.0,
        "market_cap": 190588000000.0,
    },
    {
        "symbol": "ELAN",
        "name": "Elanco Animal Health Incorporated",
        "price": 20.64,
        "change_pct": 0.83,
        "volume": 6825000.0,
        "market_cap": 10254000000.0,
    },
    {
        "symbol": "CFG",
        "name": "Citizens Financial Group, Inc.",
        "price": 53.82,
        "change_pct": 1.59,
        "volume": 6819000.0,
        "market_cap": 23215000000.0,
    },
    {
        "symbol": "CRM",
        "name": "Salesforce, Inc.",
        "price": 240.36,
        "change_pct": 0.62,
        "volume": 6817000.0,
        "market_cap": 228823000000.0,
    },
    {
        "symbol": "ENVX",
        "name": "Enovix Corporation",
        "price": 11.92,
        "change_pct": 2.32,
        "volume": 7228000.0,
        "market_cap": 2595000000.0,
    },
    {
        "symbol": "SCHW",
        "name": "The Charles Schwab Corporation",
        "price": 94.08,
        "change_pct": 1.49,
        "volume": 6761000.0,
        "market_cap": 170776000000.0,
    },
    {
        "symbol": "WRD",
        "name": "WeRide Inc.",
        "price": 10.97,
        "change_pct": -2.66,
        "volume": 6836000.0,
        "market_cap": 3121000000.0,
    },
    {
        "symbol": "NWL",
        "name": "Newell Brands Inc.",
        "price": 5.39,
        "change_pct": 3.55,
        "volume": 6726000.0,
        "market_cap": 2259000000.0,
    },
    {
        "symbol": "CL",
        "name": "Colgate-Palmolive Company",
        "price": 78.0,
        "change_pct": -0.4,
        "volume": 6723000.0,
        "market_cap": 63041000000.0,
    },
    {
        "symbol": "UAA",
        "name": "Under Armour, Inc.",
        "price": 5.05,
        "change_pct": -0.79,
        "volume": 6714000.0,
        "market_cap": 2131000000.0,
    },
    {
        "symbol": "EBAY",
        "name": "eBay Inc.",
        "price": 92.17,
        "change_pct": 4.26,
        "volume": 7537000.0,
        "market_cap": 42122000000.0,
    },
    {
        "symbol": "IPG",
        "name": "The Interpublic Group of Companies, Inc.",
        "price": 26.5,
        "change_pct": 0.91,
        "volume": 6664000.0,
        "market_cap": 9706000000.0,
    },
    {
        "symbol": "NG",
        "name": "NovaGold Resources Inc.",
        "price": 9.99,
        "change_pct": 3.52,
        "volume": 6659000.0,
        "market_cap": 4076000000.0,
    },
    {
        "symbol": "SIRI",
        "name": "Sirius XM Holdings Inc.",
        "price": 23.27,
        "change_pct": 2.99,
        "volume": 6647000.0,
        "market_cap": 7838000000.0,
    },
    {
        "symbol": "CAH",
        "name": "Cardinal Health, Inc.",
        "price": 154.46,
        "change_pct": -2.52,
        "volume": 6612000.0,
        "market_cap": 36697000000.0,
    },
    {
        "symbol": "WMB",
        "name": "The Williams Companies, Inc.",
        "price": 64.48,
        "change_pct": 0.66,
        "volume": 6551000.0,
        "market_cap": 78742000000.0,
    },
    {
        "symbol": "PPL",
        "name": "PPL Corporation",
        "price": 36.7,
        "change_pct": 0.82,
        "volume": 6529000.0,
        "market_cap": 27140000000.0,
    },
    {
        "symbol": "MDU",
        "name": "MDU Resources Group, Inc.",
        "price": 17.72,
        "change_pct": 0.23,
        "volume": 6507000.0,
        "market_cap": 3621000000.0,
    },
    {
        "symbol": "TFC",
        "name": "Truist Financial Corporation",
        "price": 45.52,
        "change_pct": 0.35,
        "volume": 6494000.0,
        "market_cap": 58695000000.0,
    },
    {
        "symbol": "AEO",
        "name": "American Eagle Outfitters, Inc.",
        "price": 16.94,
        "change_pct": -0.06,
        "volume": 6508000.0,
        "market_cap": 2869000000.0,
    },
    {
        "symbol": "GAP",
        "name": "The Gap, Inc.",
        "price": 21.59,
        "change_pct": -0.69,
        "volume": 6465000.0,
        "market_cap": 8011000000.0,
    },
    {
        "symbol": "MAT",
        "name": "Mattel, Inc.",
        "price": 18.06,
        "change_pct": 4.88,
        "volume": 6394000.0,
        "market_cap": 5819000000.0,
    },
    {
        "symbol": "STUB",
        "name": "StubHub Holdings, Inc.",
        "price": 16.95,
        "change_pct": 2.11,
        "volume": 6333000.0,
        "market_cap": 6234000000.0,
    },
    {
        "symbol": "APH",
        "name": "Amphenol Corporation",
        "price": 122.22,
        "change_pct": -1.1,
        "volume": 6315000.0,
        "market_cap": 149221000000.0,
    },
    {
        "symbol": "CNP",
        "name": "CenterPoint Energy, Inc.",
        "price": 38.86,
        "change_pct": 1.49,
        "volume": 6228000.0,
        "market_cap": 25370000000.0,
    },
    {
        "symbol": "ANET",
        "name": "Arista Networks Inc",
        "price": 145.5,
        "change_pct": 0.72,
        "volume": 6290000.0,
        "market_cap": 182874000000.0,
    },
    {
        "symbol": "MDLZ",
        "name": "Mondelez International, Inc.",
        "price": 62.67,
        "change_pct": 1.44,
        "volume": 6206000.0,
        "market_cap": 81092000000.0,
    },
    {
        "symbol": "USB",
        "name": "U.S. Bancorp",
        "price": 48.07,
        "change_pct": 0.33,
        "volume": 6184000.0,
        "market_cap": 74806000000.0,
    },
    {
        "symbol": "CRDO",
        "name": "Credo Technology Group Holding Ltd",
        "price": 143.87,
        "change_pct": -3.85,
        "volume": 6140000.0,
        "market_cap": 24889000000.0,
    },
]


def calculate_gps_score(symbol_data: dict) -> float:
    """
    Calculate GHOST Performance Score (GPS) based on market data.

    GPS Scoring Logic (0-10 scale):
    - Base score: 5.0
    - Volume boost: +0.5 if volume > avg_volume (if available)
    - Change momentum:
      - +1.0 if abs(change_pct) > 2%
      - +0.5 if abs(change_pct) > 1%
    - Volatility: +0.5 if abs(change_pct) between 0.5-5% (sweet spot)
    - Large cap stability: +0.5 if market_cap > $50B
    - Growth: +1.0 if change_pct > 3%
    - Cap at 10.0
    """
    score = 5.0

    change_pct = abs(symbol_data.get("change_pct", 0.0))
    market_cap = symbol_data.get("market_cap", 0)

    # Momentum scoring
    if change_pct > 3.0:
        score += 1.5  # Strong momentum
    elif change_pct > 2.0:
        score += 1.0  # Good momentum
    elif change_pct > 1.0:
        score += 0.5  # Moderate momentum

    # Volatility sweet spot (not too low, not too high)
    if 0.5 <= change_pct <= 5.0:
        score += 0.5

    # Large cap stability bonus
    if market_cap > 50_000_000_000:  # $50B+
        score += 0.5

    # High volume interest (if data available)
    volume = symbol_data.get("volume", 0)
    if volume > 7_000_000:  # High interest threshold
        score += 0.3

    # Cap at 10.0
    return min(10.0, score)


def populate_watchlist():
    """Populate watchlist with initial market data and GHOST scores."""
    print("🚀 Populating GHOST Watchlist...")
    print("=" * 60)

    watchlist_mgr = get_watchlist_manager()

    # Calculate scores and update watchlist
    passing_count = 0
    threshold = 7.0

    for symbol_data in INITIAL_MARKET_DATA:
        symbol = symbol_data["symbol"]
        name = symbol_data["name"]
        price = symbol_data["price"]
        change_pct = symbol_data["change_pct"]
        volume = symbol_data.get("volume")
        market_cap = symbol_data.get("market_cap")

        # Calculate GHOST score
        gps_score = calculate_gps_score(symbol_data)

        # Update score in database
        result = watchlist_mgr.update_ghost_score(
            symbol=symbol,
            gps_score=gps_score,
            price=price,
            change_pct=change_pct,
            volume=volume,
            market_cap=market_cap,
            threshold=threshold,
        )

        if result.get("passed_threshold"):
            passing_count += 1
            status = "✅ PASSED"
        else:
            status = "⏸️  WATCH"

        print(f"{status} | {symbol:6} | GPS: {gps_score:4.1f} | {change_pct:+6.2f}% | {name[:35]}")

    print("=" * 60)
    print("\n📊 Summary:")
    print(f"   Total symbols: {len(INITIAL_MARKET_DATA)}")
    print(f"   Passed threshold (GPS ≥ {threshold}): {passing_count}")
    print(f"   Pass rate: {(passing_count / len(INITIAL_MARKET_DATA) * 100):.1f}%")

    # Get statistics
    stats = watchlist_mgr.get_statistics()
    print("\n📈 Watchlist Stats:")
    print(f"   Average GPS: {stats['average_gps_score']:.2f}")
    print(f"   Symbols passing: {stats['symbols_passing_threshold']}")

    # Get top movers
    top_movers = watchlist_mgr.get_top_movers(threshold=threshold, limit=10)
    print(f"\n🔥 Top 10 Movers (GPS ≥ {threshold}):")
    for i, mover in enumerate(top_movers, 1):
        print(
            f"   {i:2}. {mover['symbol']:6} | GPS: {mover['gps']:4.1f} | {mover['change_pct']:+6.2f}% | ${mover['price']:.2f}"
        )

    print("\n✅ Watchlist population complete!")
    print("\n💡 Tip: These symbols are now in your watchlist.")
    print(f"   Only symbols with GPS ≥ {threshold} will appear in /api/top_movers")
    print(f"   This is your buy signal list - when GPS passes {threshold}, consider buying!\n")


if __name__ == "__main__":
    populate_watchlist()
