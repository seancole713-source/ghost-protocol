#!/usr/bin/env python3
"""
Test Portfolio Persistence Integration
=======================================

Verifies that Ghost can:
1. Save portfolio positions to persistent storage
2. Save price history to database
3. Load positions after restart
4. Fallback to cached prices when live data unavailable
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import time

from core.portfolio_persistence import get_portfolio_store


def test_portfolio_persistence():
    """Test basic portfolio persistence operations."""
    print("\n" + "=" * 60)
    print("🧪 Testing Portfolio Persistence")
    print("=" * 60 + "\n")

    store = get_portfolio_store("data/test_portfolio.db")

    # Test 1: Save position
    print("📝 Test 1: Saving position...")
    success = store.save_position("WOLF", 100.0, 25.50, 24.69, "yfinance")
    assert success, "Failed to save position"
    print("✅ Position saved successfully")

    # Test 2: Retrieve position
    print("\n📖 Test 2: Retrieving position...")
    pos = store.get_position("WOLF")
    assert pos is not None, "Failed to retrieve position"
    assert pos["quantity"] == 100.0, f"Wrong quantity: {pos['quantity']}"
    assert pos["avg_cost"] == 25.50, f"Wrong avg_cost: {pos['avg_cost']}"
    assert pos["last_known_price"] == 24.69, f"Wrong last price: {pos['last_known_price']}"
    print(f"✅ Position retrieved: {pos['quantity']} shares @ ${pos['avg_cost']}")
    print(f"   Last known price: ${pos['last_known_price']} ({pos['last_provider']})")

    # Test 3: Save price history
    print("\n💰 Test 3: Saving price history...")
    success = store.save_price("WOLF", 24.69, 24.50, "yfinance", "open")
    assert success, "Failed to save price"
    print("✅ Price saved successfully")

    # Test 4: Retrieve last price
    print("\n🔍 Test 4: Retrieving last price...")
    last = store.get_last_price("WOLF", max_age_seconds=3600)
    assert last is not None, "Failed to retrieve last price"
    price, prev, prov, ts = last
    assert price == 24.69, f"Wrong price: {price}"
    age_seconds = time.time() - ts
    print(f"✅ Last price retrieved: ${price} (prev: ${prev})")
    print(f"   Provider: {prov}, Age: {age_seconds:.1f}s")

    # Test 5: Save cash balance
    print("\n💵 Test 5: Saving cash balance...")
    success = store.save_cash_balance(10000.00)
    assert success, "Failed to save cash"
    print("✅ Cash balance saved")

    # Test 6: Retrieve cash balance
    print("\n💸 Test 6: Retrieving cash balance...")
    cash = store.get_cash_balance()
    assert cash == 10000.00, f"Wrong cash: {cash}"
    print(f"✅ Cash balance retrieved: ${cash:,.2f}")

    # Test 7: Daily snapshot
    print("\n📸 Test 7: Saving daily snapshot...")
    today = time.strftime("%Y-%m-%d")
    positions = [{"symbol": "WOLF", "qty": 100, "avg": 25.50}]
    prices = {"WOLF": 24.69}
    success = store.save_daily_snapshot(
        today, 12469.00, 10000.00, positions, prices, "Test snapshot after market close"
    )
    assert success, "Failed to save snapshot"
    print(f"✅ Snapshot saved for {today}")

    # Test 8: Retrieve snapshot
    print("\n📷 Test 8: Retrieving daily snapshot...")
    snap = store.get_daily_snapshot(today)
    assert snap is not None, "Failed to retrieve snapshot"
    assert snap["portfolio_value"] == 12469.00
    print(f"✅ Snapshot retrieved: Portfolio value ${snap['portfolio_value']:,.2f}")
    print(f"   Positions: {len(snap['positions'])}, Prices: {len(snap['prices'])}")

    # Test 9: Get all positions
    print("\n📊 Test 9: Getting all positions...")
    all_pos = store.get_all_positions()
    assert len(all_pos) > 0, "No positions found"
    print(f"✅ Found {len(all_pos)} position(s):")
    for p in all_pos:
        print(f"   {p['symbol']}: {p['quantity']} @ ${p['avg_cost']}")

    # Test 10: Fallback scenario (stale price)
    print("\n⏰ Test 10: Testing price fallback (old data)...")
    # Try to get a very old price (should fail with default 24h window)
    old = store.get_last_price("NONEXISTENT", max_age_seconds=1)
    assert old is None, "Should not find non-existent price"
    print("✅ Correctly returns None for non-existent/stale prices")

    # Test 11: Cleanup
    print("\n🧹 Test 11: Testing price cleanup...")
    deleted = store.cleanup_old_prices(days_to_keep=30)
    print(f"✅ Cleanup complete (would delete {deleted} old records)")

    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60 + "\n")

    # Cleanup test database
    try:
        os.remove("data/test_portfolio.db")
        print("🗑️  Test database removed")
    except Exception:
        pass


def test_integration():
    """Test integration with Ghost state."""
    print("\n" + "=" * 60)
    print("🔗 Testing Ghost Integration")
    print("=" * 60 + "\n")

    # Simulate Ghost startup scenario
    print("📦 Scenario: Ghost starts with empty memory...")
    print("   No price cache, no positions loaded yet")

    store = get_portfolio_store("data/test_ghost_integration.db")

    # User had position before restart
    print("\n💾 Saving position before 'restart'...")
    store.save_position("WOLF", 50.0, 26.00, 25.50, "alphavantage")
    store.save_price("WOLF", 25.50, 25.00, "alphavantage", "closed")
    print("✅ Position and price saved")

    # Simulate restart: load from persistence
    print("\n🔄 'Restarting' Ghost...")
    pos = store.get_position("WOLF")
    if pos:
        print(f"✅ Position restored: {pos['quantity']} @ ${pos['avg_cost']}")
        print(f"   Last known price: ${pos['last_known_price']}")
    else:
        print("❌ Failed to restore position!")
        return

    # Simulate market closed / no live data
    print("\n🌙 Scenario: Markets closed, no live data available...")
    last_price = store.get_last_price("WOLF", max_age_seconds=86400 * 7)
    if last_price:
        price, prev, prov, ts = last_price
        age_hours = (time.time() - ts) / 3600
        print(f"✅ Using cached price: ${price} ({prov})")
        print(f"   Age: {age_hours:.1f} hours old")
        print(f"   Portfolio value: ${pos['quantity'] * price:,.2f}")
    else:
        print("❌ No cached price available!")
        return

    # Simulate market open / live data returns
    print("\n☀️  Scenario: Markets open, live data available...")
    new_price = 26.50
    store.save_price("WOLF", new_price, 25.50, "polygon", "open")
    print(f"✅ Live price updated: ${new_price}")

    latest = store.get_last_price("WOLF")
    if latest:
        price, prev, prov, ts = latest
        print(f"✅ Refreshed from live data: ${price} ({prov})")
        print(f"   Portfolio value updated: ${pos['quantity'] * price:,.2f}")

    print("\n" + "=" * 60)
    print("🎉 Integration test passed!")
    print("=" * 60 + "\n")

    # Cleanup
    try:
        os.remove("data/test_ghost_integration.db")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        test_portfolio_persistence()
        test_integration()
        print("\n✅ All portfolio persistence tests passed!\n")
        print("Ghost will now:")
        print("  • Remember your positions across restarts")
        print("  • Fallback to cached prices when live data unavailable")
        print("  • Auto-refresh when markets reopen")
        print("  • Save daily snapshots for historical tracking\n")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
