#!/usr/bin/env python3
"""
Test Polygon Historical Price Integration
Validates that the new _get_price_at_time() function can fetch historical prices
"""

import os
import sys
import time
from datetime import datetime, timedelta

# Set up environment
os.environ["POLYGON_API_KEY"] = os.getenv("POLYGON_API_KEY", "8VIvELVXiLG30K2l1348RzSurffLM0jR")

# Import the function
sys.path.insert(0, "/workspaces/ghost-protocol")
from services.outcome_reconciler_v2 import _get_price_at_time


def test_historical_fetch():
    """Test fetching historical prices at different time offsets"""
    
    print("=" * 70)
    print("TESTING POLYGON HISTORICAL PRICE INTEGRATION")
    print("=" * 70)
    
    # Test cases: symbol, hours_back
    test_cases = [
        ("AAPL", 1),     # 1 hour ago (should use live price fallback)
        ("AAPL", 6),     # 6 hours ago (should use Polygon minute bars)
        ("AAPL", 24),    # 24 hours ago (should use Polygon)
        ("AAPL", 48),    # 48 hours ago (Ghost's reconciliation window)
        ("BTC", 48),     # Crypto (may fail - Polygon crypto requires different endpoint)
        ("TSLA", 72),    # 3 days ago
    ]
    
    results = []
    
    for symbol, hours_back in test_cases:
        # Calculate target timestamp
        target_time = time.time() - (hours_back * 3600)
        target_dt = datetime.fromtimestamp(target_time)
        
        print(f"\n{'─' * 70}")
        print(f"TEST: {symbol} @ {hours_back}h ago ({target_dt.strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"{'─' * 70}")
        
        try:
            # Fetch price
            start = time.time()
            price = _get_price_at_time(symbol, target_time)
            elapsed = time.time() - start
            
            if price is not None:
                print(f"✅ SUCCESS: ${price:.2f} (fetched in {elapsed:.2f}s)")
                results.append({
                    "symbol": symbol,
                    "hours_back": hours_back,
                    "price": price,
                    "elapsed": elapsed,
                    "status": "success"
                })
            else:
                print(f"⚠️  NO DATA: Could not fetch historical price")
                results.append({
                    "symbol": symbol,
                    "hours_back": hours_back,
                    "price": None,
                    "elapsed": elapsed,
                    "status": "no_data"
                })
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                "symbol": symbol,
                "hours_back": hours_back,
                "price": None,
                "elapsed": 0,
                "status": "error",
                "error": str(e)
            })
    
    # Summary
    print(f"\n{'=' * 70}")
    print("TEST RESULTS SUMMARY")
    print(f"{'=' * 70}")
    
    success_count = sum(1 for r in results if r["status"] == "success")
    no_data_count = sum(1 for r in results if r["status"] == "no_data")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"✅ Success: {success_count}/{len(results)}")
    print(f"⚠️  No Data: {no_data_count}/{len(results)}")
    print(f"❌ Errors: {error_count}/{len(results)}")
    
    if success_count > 0:
        avg_time = sum(r["elapsed"] for r in results if r["status"] == "success") / success_count
        print(f"\nAverage Fetch Time: {avg_time:.2f}s")
    
    # Detailed results
    print(f"\n{'─' * 70}")
    print("DETAILED RESULTS")
    print(f"{'─' * 70}")
    for r in results:
        status_icon = {"success": "✅", "no_data": "⚠️", "error": "❌"}[r["status"]]
        price_str = f"${r['price']:.2f}" if r["price"] else "N/A"
        print(f"{status_icon} {r['symbol']:6s} | {r['hours_back']:3d}h ago | {price_str:10s} | {r['elapsed']:.2f}s")
    
    print(f"\n{'=' * 70}")
    
    # Final verdict
    if success_count >= 4:  # At least 4/6 should work (crypto might fail)
        print("✅ POLYGON INTEGRATION: READY FOR PRODUCTION")
        return 0
    else:
        print("❌ POLYGON INTEGRATION: NEEDS DEBUGGING")
        return 1


if __name__ == "__main__":
    try:
        exit_code = test_historical_fetch()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
