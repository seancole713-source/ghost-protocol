#!/usr/bin/env python3
"""
Test Personal Watchlist Deployment on Railway Production
=========================================================

This script tests the personal watchlist API endpoints on Railway production.
Run after deployment and migration to verify everything works.

Usage:
    python3 test_production_watchlist.py
    python3 test_production_watchlist.py --base-url https://your-domain.railway.app
"""

import argparse
import json
import sys
import time

import pytest
import requests

pytestmark = pytest.mark.skip(reason="Manual integration script — hits production Railway deployment")

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def test_get_user_watchlist(base_url: str) -> bool:
    """Test GET /api/v3/watchlist/user"""
    print(f"\n{BLUE}Test 1: GET /api/v3/watchlist/user{RESET}")
    try:
        response = requests.get(f"{base_url}/api/v3/watchlist/user", timeout=10)
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Response: {json.dumps(data, indent=2)}")
            print(f"  {GREEN}✅ PASS{RESET}")
            return True
        else:
            print(f"  {RED}❌ FAIL: Expected 200, got {response.status_code}{RESET}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"  {RED}❌ FAIL: {e}{RESET}")
        return False


def test_get_stats(base_url: str) -> bool:
    """Test GET /api/v3/watchlist/stats"""
    print(f"\n{BLUE}Test 2: GET /api/v3/watchlist/stats{RESET}")
    try:
        response = requests.get(f"{base_url}/api/v3/watchlist/stats", timeout=10)
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Response: {json.dumps(data, indent=2)}")
            print(f"  {GREEN}✅ PASS{RESET}")
            return True
        else:
            print(f"  {RED}❌ FAIL: Expected 200, got {response.status_code}{RESET}")
            return False
    except Exception as e:
        print(f"  {RED}❌ FAIL: {e}{RESET}")
        return False


def test_add_symbol(base_url: str, symbol: str, asset_type: str) -> bool:
    """Test POST /api/v3/watchlist/add"""
    print(f"\n{BLUE}Test 3: POST /api/v3/watchlist/add (symbol={symbol}, type={asset_type}){RESET}")
    try:
        payload = {
            "symbol": symbol,
            "asset_type": asset_type,
            "owns_position": False,
            "notes": "Test symbol added via production test script",
            "alert_threshold_pct": 5.0,
            "priority": 1
        }
        response = requests.post(f"{base_url}/api/v3/watchlist/add", json=payload, timeout=10)
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Response: {json.dumps(data, indent=2)}")
            if data.get("ok"):
                print(f"  {GREEN}✅ PASS{RESET}")
                return True
            else:
                print(f"  {YELLOW}⚠️  API returned ok=False: {data.get('error')}{RESET}")
                return False
        else:
            print(f"  {RED}❌ FAIL: Expected 200, got {response.status_code}{RESET}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"  {RED}❌ FAIL: {e}{RESET}")
        return False


def test_update_position(base_url: str, symbol: str, asset_type: str) -> bool:
    """Test POST /api/v3/watchlist/update-position"""
    print(f"\n{BLUE}Test 4: POST /api/v3/watchlist/update-position (toggle ownership){RESET}")
    try:
        payload = {
            "symbol": symbol,
            "asset_type": asset_type,
            "owns_position": True
        }
        response = requests.post(f"{base_url}/api/v3/watchlist/update-position", json=payload, timeout=10)
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Response: {json.dumps(data, indent=2)}")
            if data.get("ok"):
                print(f"  {GREEN}✅ PASS{RESET}")
                return True
            else:
                print(f"  {YELLOW}⚠️  API returned ok=False: {data.get('error')}{RESET}")
                return False
        else:
            print(f"  {RED}❌ FAIL: Expected 200, got {response.status_code}{RESET}")
            return False
    except Exception as e:
        print(f"  {RED}❌ FAIL: {e}{RESET}")
        return False


def test_get_history(base_url: str, symbol: str) -> bool:
    """Test GET /api/v3/watchlist/history/{symbol}"""
    print(f"\n{BLUE}Test 5: GET /api/v3/watchlist/history/{symbol}{RESET}")
    try:
        response = requests.get(f"{base_url}/api/v3/watchlist/history/{symbol}?limit=5", timeout=10)
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Response: {json.dumps(data, indent=2)[:500]}...")
            print(f"  {GREEN}✅ PASS{RESET}")
            return True
        else:
            print(f"  {RED}❌ FAIL: Expected 200, got {response.status_code}{RESET}")
            return False
    except Exception as e:
        print(f"  {RED}❌ FAIL: {e}{RESET}")
        return False


def test_remove_symbol(base_url: str, symbol: str, asset_type: str) -> bool:
    """Test POST /api/v3/watchlist/remove"""
    print(f"\n{BLUE}Test 6: POST /api/v3/watchlist/remove (cleanup){RESET}")
    try:
        payload = {
            "symbol": symbol,
            "asset_type": asset_type
        }
        response = requests.post(f"{base_url}/api/v3/watchlist/remove", json=payload, timeout=10)
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Response: {json.dumps(data, indent=2)}")
            if data.get("ok"):
                print(f"  {GREEN}✅ PASS{RESET}")
                return True
            else:
                print(f"  {YELLOW}⚠️  API returned ok=False: {data.get('error')}{RESET}")
                return False
        else:
            print(f"  {RED}❌ FAIL: Expected 200, got {response.status_code}{RESET}")
            return False
    except Exception as e:
        print(f"  {RED}❌ FAIL: {e}{RESET}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Personal Watchlist on Railway Production")
    parser.add_argument(
        "--base-url",
        default="https://ghost-protocol-production.up.railway.app",
        help="Base URL of Railway deployment"
    )
    parser.add_argument(
        "--test-symbol",
        default="TESTAPI",
        help="Symbol to use for testing (will be added and removed)"
    )
    parser.add_argument(
        "--asset-type",
        default="stock",
        choices=["stock", "crypto"],
        help="Asset type for test symbol"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Personal Watchlist Production Test Suite")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")
    print(f"Test Symbol: {args.test_symbol} ({args.asset_type})")
    print()

    results = []

    # Test 1: Get user watchlist (should work even if empty)
    results.append(("GET /user", test_get_user_watchlist(args.base_url)))
    time.sleep(1)

    # Test 2: Get stats
    results.append(("GET /stats", test_get_stats(args.base_url)))
    time.sleep(1)

    # Test 3: Add symbol
    results.append(("POST /add", test_add_symbol(args.base_url, args.test_symbol, args.asset_type)))
    time.sleep(1)

    # Test 4: Update position flag
    results.append(("POST /update-position", test_update_position(args.base_url, args.test_symbol, args.asset_type)))
    time.sleep(1)

    # Test 5: Get history
    results.append(("GET /history", test_get_history(args.base_url, args.test_symbol)))
    time.sleep(1)

    # Test 6: Remove symbol (cleanup)
    results.append(("POST /remove", test_remove_symbol(args.base_url, args.test_symbol, args.asset_type)))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{GREEN}✅ PASS{RESET}" if result else f"{RED}❌ FAIL{RESET}"
        print(f"  {test_name:30s} {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n{GREEN}✅ ALL TESTS PASSED!{RESET}")
        print("\nPersonal watchlist is working correctly on production! 🎉")
        sys.exit(0)
    else:
        print(f"\n{RED}❌ SOME TESTS FAILED{RESET}")
        print("\nCheck the errors above and verify:")
        print("  1. Database migration ran successfully")
        print("  2. Wolf_app.py includes watchlist router")
        print("  3. Railway service restarted after deployment")
        sys.exit(1)


if __name__ == "__main__":
    main()
