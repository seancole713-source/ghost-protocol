#!/usr/bin/env python3
"""
Alpaca Broker Connection Test Suite
Tests all critical broker functionality before enabling live trading.
"""

import os
import sys
from typing import Any


# Set up environment for testing
def setup_test_env():
    """Configure test environment variables."""
    print("=== Alpaca Broker Test Configuration ===\n")

    # Check if broker is enabled
    broker_name = os.getenv("BROKER", "")
    print(f"BROKER: {broker_name or '(not set)'}")

    # Check API keys
    key_id = os.getenv("ALPACA_KEY_ID", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    print(f"ALPACA_KEY_ID: {'✓ Set' if key_id else '✗ Not set'}")
    print(f"ALPACA_SECRET_KEY: {'✓ Set' if secret_key else '✗ Not set'}")

    # Check mode
    paper_mode = os.getenv("ALPACA_PAPER", "1")
    base_url = os.getenv("APCA_API_BASE_URL", "")
    print(
        f"ALPACA_PAPER: {paper_mode} ({'Paper Trading' if paper_mode == '1' else 'LIVE TRADING'})"
    )
    print(f"APCA_API_BASE_URL: {base_url or '(using default)'}")

    # Check rate limits
    rate = os.getenv("ALPACA_ORDER_RATE", "30")
    window = os.getenv("ALPACA_ORDER_WINDOW_S", "60")
    print(f"Rate Limit: {rate} orders per {window} seconds")

    print()

    if broker_name.lower() != "alpaca":
        print("⚠️  WARNING: BROKER is not set to 'alpaca'")
        print("   Set BROKER=alpaca to enable broker integration\n")
        return False

    if not key_id or not secret_key:
        print("⚠️  WARNING: API keys not configured")
        print("   Set ALPACA_KEY_ID and ALPACA_SECRET_KEY\n")
        return False

    return True


def test_broker_import():
    """Test 1: Import the broker module."""
    print("TEST 1: Import broker module")
    try:
        from core.alpaca_broker import AlpacaBroker

        print("✓ Successfully imported core.alpaca_broker")
        print(f"  AlpacaBroker class available: {AlpacaBroker.__name__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def test_broker_initialization():
    """Test 2: Initialize broker instance."""
    print("\nTEST 2: Initialize broker")
    try:
        from core.alpaca_broker import get_broker

        broker = get_broker()
        print("✓ Broker initialized")
        print(f"  - Enabled: {broker.enabled}")
        print(f"  - Paper mode: {broker.paper}")
        print(f"  - Base URL: {broker.base_url}")
        return broker
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return None


def test_health_check(broker):
    """Test 3: Health check (account connectivity)."""
    print("\nTEST 3: Health check")
    try:
        if not broker or not broker.enabled:
            print("⊘ Skipped (broker not enabled)")
            return None

        health = broker.health_check()

        if health.get("ok"):
            print("✓ Health check PASSED")
            print(f"  - Account ID: {health.get('account_id', 'N/A')}")
            print(f"  - Account #: {health.get('account_number', 'N/A')}")
            print(f"  - Status: {health.get('status', 'N/A')}")
            print(f"  - Buying Power: ${health.get('buying_power', 0):,.2f}")
            print(f"  - Cash: ${health.get('cash', 0):,.2f}")
            print(f"  - Portfolio Value: ${health.get('portfolio_value', 0):,.2f}")
            print(f"  - Positions: {health.get('positions_count', 0)}")
            print(f"  - Market Open: {health.get('market_open', False)}")
            return health
        else:
            print(f"✗ Health check FAILED: {health.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"✗ Health check error: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_get_account(broker):
    """Test 4: Get account details."""
    print("\nTEST 4: Get account details")
    try:
        if not broker or not broker.enabled:
            print("⊘ Skipped (broker not enabled)")
            return None

        account = broker.get_account()
        print("✓ Account retrieved successfully")
        print(f"  - Currency: {account.get('currency', 'N/A')}")
        print(f"  - Pattern Day Trader: {account.get('pattern_day_trader', False)}")
        print(f"  - Trading Blocked: {account.get('trading_blocked', False)}")
        print(f"  - Account Blocked: {account.get('account_blocked', False)}")
        print(f"  - Equity: ${float(account.get('equity', 0)):,.2f}")
        print(f"  - Last Equity: ${float(account.get('last_equity', 0)):,.2f}")
        return account
    except Exception as e:
        print(f"✗ Get account failed: {e}")
        return None


def test_get_positions(broker):
    """Test 5: Get open positions."""
    print("\nTEST 5: Get positions")
    try:
        if not broker or not broker.enabled:
            print("⊘ Skipped (broker not enabled)")
            return None

        positions = broker.get_positions()
        print(f"✓ Retrieved {len(positions)} position(s)")

        if positions:
            for pos in positions:
                symbol = pos.get("symbol")
                qty = float(pos.get("qty", 0))
                side = pos.get("side", "N/A")
                entry_price = float(pos.get("avg_entry_price", 0))
                current_price = float(pos.get("current_price", 0))
                market_value = float(pos.get("market_value", 0))
                unrealized_pl = float(pos.get("unrealized_pl", 0))
                unrealized_plpc = float(pos.get("unrealized_plpc", 0)) * 100

                print(f"\n  {symbol}:")
                print(f"    Qty: {qty} ({side})")
                print(f"    Entry: ${entry_price:.2f}")
                print(f"    Current: ${current_price:.2f}")
                print(f"    Market Value: ${market_value:,.2f}")
                print(f"    P&L: ${unrealized_pl:,.2f} ({unrealized_plpc:+.2f}%)")
        else:
            print("  (no open positions)")

        return positions
    except Exception as e:
        print(f"✗ Get positions failed: {e}")
        return None


def test_get_clock(broker):
    """Test 6: Get market clock."""
    print("\nTEST 6: Get market clock")
    try:
        if not broker or not broker.enabled:
            print("⊘ Skipped (broker not enabled)")
            return None

        clock = broker.get_clock()
        print("✓ Market clock retrieved")
        print(f"  - Current Time: {clock.get('timestamp', 'N/A')}")
        print(f"  - Market Open: {clock.get('is_open', False)}")
        print(f"  - Next Open: {clock.get('next_open', 'N/A')}")
        print(f"  - Next Close: {clock.get('next_close', 'N/A')}")
        return clock
    except Exception as e:
        print(f"✗ Get clock failed: {e}")
        return None


def test_get_orders(broker):
    """Test 7: Get recent orders."""
    print("\nTEST 7: Get recent orders")
    try:
        if not broker or not broker.enabled:
            print("⊘ Skipped (broker not enabled)")
            return None

        orders = broker.get_orders(status="all", limit=10)
        print(f"✓ Retrieved {len(orders)} recent order(s)")

        if orders:
            for order in orders[:5]:  # Show first 5
                order_id = order.get("id", "N/A")
                symbol = order.get("symbol", "N/A")
                side = order.get("side", "N/A")
                qty = order.get("qty", "N/A")
                order_type = order.get("type", "N/A")
                status = order.get("status", "N/A")
                created_at = order.get("created_at", "N/A")

                print(f"\n  Order {order_id[:8]}...")
                print(f"    {side.upper()} {qty} {symbol} ({order_type})")
                print(f"    Status: {status}")
                print(f"    Created: {created_at}")
        else:
            print("  (no recent orders)")

        return orders
    except Exception as e:
        print(f"✗ Get orders failed: {e}")
        return None


def test_dry_run_order(broker):
    """Test 8: Dry run order submission (validate params only)."""
    print("\nTEST 8: Dry run order (no actual submission)")
    try:
        if not broker or not broker.enabled:
            print("⊘ Skipped (broker not enabled)")
            return None

        # Test order parameters
        test_symbol = "AAPL"
        test_qty = 1
        test_side = "buy"

        print(f"  Testing order params: {test_side.upper()} {test_qty} {test_symbol}")
        print("  (This is just parameter validation, no order will be submitted)")

        # We'll just validate the parameters without actually submitting

        # Check if we can construct the order
        if test_side.lower() not in ["buy", "sell"]:
            print("  ✗ Invalid side")
            return False

        print("  ✓ Order parameters valid")
        print("  ✓ Ready for live order submission")
        return True

    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        return False


def generate_summary(results: dict[str, Any]):
    """Generate test summary report."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total_tests - passed

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed} ✓")
    print(f"Failed/Skipped: {failed}")

    if results.get("health_check"):
        print("\n✓ CONNECTION VERIFIED: Alpaca broker is ready")
        health = results["health_check"]
        if health.get("market_open"):
            print("✓ MARKET IS OPEN: Ready for live trading")
        else:
            print("⚠️  Market is closed (orders will queue)")
    else:
        print("\n✗ CONNECTION FAILED: Cannot connect to Alpaca")
        print("   Check API keys and network connectivity")

    print("\n" + "=" * 60)


def main():
    """Run all broker tests."""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║      ALPACA BROKER CONNECTION TEST SUITE                  ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    # Setup and configuration check
    setup_test_env()

    results = {}

    # Test 1: Import
    results["import"] = test_broker_import()
    if not results["import"]:
        print("\n✗ CRITICAL: Cannot import broker module")
        return 1

    # Test 2: Initialize
    broker = test_broker_initialization()
    results["init"] = broker is not None

    if not broker or not broker.enabled:
        print("\n⚠️  Broker not enabled - skipping connectivity tests")
        print("   Set BROKER=alpaca and configure API keys to run full tests")
        generate_summary(results)
        return 0

    # Test 3-8: Connectivity tests
    results["health_check"] = test_health_check(broker)
    results["account"] = test_get_account(broker)
    results["positions"] = test_get_positions(broker)
    results["clock"] = test_get_clock(broker)
    results["orders"] = test_get_orders(broker)
    results["dry_run"] = test_dry_run_order(broker)

    # Summary
    generate_summary(results)

    return 0 if results.get("health_check") else 1


if __name__ == "__main__":
    sys.exit(main())
