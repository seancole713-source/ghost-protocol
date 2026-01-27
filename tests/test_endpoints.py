#!/usr/bin/env python3
"""
Comprehensive endpoint test for Ghost Protocol
Tests both local turbo provider and Railway production endpoints
"""

import sys
import time

sys.path.insert(0, '.')

from core.providers.turbo_provider import TurboProvider


def test_local_turbo():
    """Test turbo provider locally"""
    print("=" * 60)
    print("LOCAL TURBO PROVIDER TESTS")
    print("=" * 60)

    turbo = TurboProvider()

    # Test BTC
    print("\n1. Testing BTC (crypto)...")
    start = time.time()
    btc = turbo.turbo_crypto_price('BTC', max_budget_s=4.0)
    duration = time.time() - start

    if btc['ok']:
        print("   ✅ SUCCESS")
        print(f"   Price: ${btc['price']:,.2f}")
        print(f"   Provider: {btc['provider']}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Cached: {btc.get('cached', False)}")
    else:
        print("   ❌ FAILED")
        print(f"   Error: {btc.get('error')}")
        print(f"   Logs: {btc.get('logs')}")

    # Test PACS
    print("\n2. Testing PACS (stock)...")
    start = time.time()
    pacs = turbo.turbo_stock_price('PACS', max_budget_s=4.0)
    duration = time.time() - start

    if pacs['ok']:
        print("   ✅ SUCCESS")
        print(f"   Price: ${pacs['price']:.2f}")
        print(f"   Provider: {pacs['provider']}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Cached: {pacs.get('cached', False)}")
    else:
        print("   ❌ FAILED")
        print(f"   Error: {pacs.get('error')}")
        print(f"   Logs: {pacs.get('logs')}")

    # Summary
    print("\n" + "=" * 60)
    print("LOCAL TEST SUMMARY")
    print("=" * 60)
    btc_status = "✅ PASS" if btc['ok'] else "❌ FAIL"
    pacs_status = "✅ PASS" if pacs['ok'] else "❌ FAIL"
    print(f"BTC:  {btc_status} ({btc.get('provider', 'N/A')}, {btc.get('price', 'N/A')})")
    print(f"PACS: {pacs_status} ({pacs.get('provider', 'N/A')}, {pacs.get('price', 'N/A')})")

    return btc['ok'] and pacs['ok']


if __name__ == "__main__":
    success = test_local_turbo()
    sys.exit(0 if success else 1)
