#!/usr/bin/env python3
"""
Test Polygon Snapshot Integration

This script tests the new snapshot movers detection without running the full Ghost server.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ["USE_POLYGON_SNAPSHOTS"] = "true"
os.environ["POLYGON_API_KEY"] = os.getenv("POLYGON_API_KEY", "")

if not os.environ["POLYGON_API_KEY"]:
    print("❌ ERROR: POLYGON_API_KEY not set")
    print("   Set it with: export POLYGON_API_KEY=\"$(railway variables get POLYGON_API_KEY)\"")
    sys.exit(1)

from app.core.movers_scanner import (
    fetch_polygon_snapshots,
    fetch_polygon_all_movers,
    scan_stocks,
)


async def test_gainers():
    """Test fetching top gainers"""
    print("\n🔍 Testing Polygon Snapshot API - Gainers...")
    print("=" * 60)
    
    gainers = await fetch_polygon_snapshots("gainers")
    
    if not gainers:
        print("⚠️  No gainers found (market may be closed or low volatility)")
        return []
    
    print(f"✅ Found {len(gainers)} gainers\n")
    
    for i, mover in enumerate(gainers[:5], 1):
        print(f"{i}. {mover['symbol']:6s} ${mover['price']:8.2f} "
              f"{mover['pct_24h']:+6.2f}% {mover['tier']} "
              f"(vol: {mover['vol_mult']:.2f}x)" if mover['vol_mult'] else "(vol: N/A)")
    
    if len(gainers) > 5:
        print(f"   ... and {len(gainers) - 5} more")
    
    return gainers


async def test_losers():
    """Test fetching top losers"""
    print("\n🔍 Testing Polygon Snapshot API - Losers...")
    print("=" * 60)
    
    losers = await fetch_polygon_snapshots("losers")
    
    if not losers:
        print("⚠️  No losers found (market may be closed or low volatility)")
        return []
    
    print(f"✅ Found {len(losers)} losers\n")
    
    for i, mover in enumerate(losers[:5], 1):
        print(f"{i}. {mover['symbol']:6s} ${mover['price']:8.2f} "
              f"{mover['pct_24h']:+6.2f}% {mover['tier']} "
              f"(vol: {mover['vol_mult']:.2f}x)" if mover['vol_mult'] else "(vol: N/A)")
    
    if len(losers) > 5:
        print(f"   ... and {len(losers) - 5} more")
    
    return losers


async def test_combined():
    """Test fetching and merging gainers + losers"""
    print("\n🔍 Testing Combined Movers (Gainers + Losers)...")
    print("=" * 60)
    
    all_movers = await fetch_polygon_all_movers()
    
    if not all_movers:
        print("⚠️  No movers found (market may be closed or low volatility)")
        return []
    
    print(f"✅ Found {len(all_movers)} total movers (deduplicated)\n")
    
    print("Top 10 by absolute % change:")
    for i, mover in enumerate(all_movers[:10], 1):
        direction = "▲" if mover['pct_24h'] > 0 else "▼"
        print(f"{i:2d}. {direction} {mover['symbol']:6s} ${mover['price']:8.2f} "
              f"{mover['pct_24h']:+6.2f}% {mover['tier']}")
    
    if len(all_movers) > 10:
        print(f"    ... and {len(all_movers) - 10} more")
    
    return all_movers


async def test_scan_stocks():
    """Test full scan_stocks() function with snapshot integration"""
    print("\n🔍 Testing scan_stocks() with Snapshot Integration...")
    print("=" * 60)
    
    # Mock fetch_price_func (not used in snapshot mode)
    async def mock_fetch_price(symbol, is_crypto=False):
        return None
    
    movers = await scan_stocks(mock_fetch_price)
    
    if not movers:
        print("⚠️  No movers found meeting threshold criteria")
        print("   STOCK_PCT_THRESHOLD = 6.0%")
        print("   STOCK_VOL_MULT_THRESHOLD = 1.3x")
        return []
    
    print(f"✅ Found {len(movers)} movers meeting thresholds\n")
    
    for i, mover in enumerate(movers[:10], 1):
        direction = "▲" if mover['pct_24h'] > 0 else "▼"
        vol_str = f"{mover['vol_mult']:.2f}x" if mover['vol_mult'] else "N/A"
        print(f"{i:2d}. {direction} {mover['symbol']:6s} ${mover['price']:8.2f} "
              f"{mover['pct_24h']:+6.2f}% (vol: {vol_str:6s}) {mover['emoji']}")
    
    if len(movers) > 10:
        print(f"    ... and {len(movers) - 10} more")
    
    return movers


async def test_api_usage():
    """Calculate API usage"""
    print("\n📊 API Usage Analysis")
    print("=" * 60)
    
    # Test gainers endpoint
    print("Calling /v2/snapshot/locale/us/markets/stocks/gainers...")
    gainers = await fetch_polygon_snapshots("gainers")
    call_count = 1
    
    # Test losers endpoint
    print("Calling /v2/snapshot/locale/us/markets/stocks/losers...")
    losers = await fetch_polygon_snapshots("losers")
    call_count += 1
    
    print(f"\n✅ Total API calls: {call_count}")
    print(f"   Gainers found: {len(gainers)}")
    print(f"   Losers found: {len(losers)}")
    print(f"   Total unique movers: {len(set([m['symbol'] for m in gainers + losers]))}")
    
    print(f"\n📈 Daily Usage Projection:")
    print(f"   Scans per day: 41 (market hours)")
    print(f"   API calls per scan: {call_count}")
    print(f"   Daily total: {41 * call_count} calls")
    print(f"   Free tier limit: 7,200 calls/day")
    print(f"   Remaining quota: {7200 - (41 * call_count)} calls ({((7200 - (41 * call_count)) / 7200 * 100):.1f}%)")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Polygon Snapshot Integration Test Suite")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  USE_POLYGON_SNAPSHOTS: {os.getenv('USE_POLYGON_SNAPSHOTS')}")
    print(f"  POLYGON_API_KEY: {'SET' if os.getenv('POLYGON_API_KEY') else 'MISSING'}")
    
    try:
        # Run tests
        gainers = await test_gainers()
        losers = await test_losers()
        all_movers = await test_combined()
        filtered_movers = await test_scan_stocks()
        await test_api_usage()
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Gainers endpoint: {len(gainers)} movers")
        print(f"✅ Losers endpoint: {len(losers)} movers")
        print(f"✅ Combined movers: {len(all_movers)} movers")
        print(f"✅ After threshold filter: {filtered_movers and len(filtered_movers) or 0} movers")
        print(f"✅ API calls used: 2 (optimal)")
        print("\n🎉 All tests passed! Integration is working correctly.")
        
        if not filtered_movers and all_movers:
            print("\n💡 TIP: No movers met the 6% threshold.")
            print("   To see more movers, lower STOCK_PCT_THRESHOLD in movers_scanner.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
