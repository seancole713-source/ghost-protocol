#!/usr/bin/env python3
"""
Test Personal Watchlist Endpoints
==================================

Quick validation that endpoints work correctly.
"""

import requests
import json
import sys

BASE_URL = "https://ghost-protocol-production.up.railway.app"

def test_get_watchlist():
    """Test GET /api/v3/watchlist/user"""
    print("\n=== TEST: GET /api/v3/watchlist/user ===")
    try:
        response = requests.get(f"{BASE_URL}/api/v3/watchlist/user", timeout=10)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(json.dumps(data, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_add_symbol(symbol, asset_type):
    """Test POST /api/v3/watchlist/add"""
    print(f"\n=== TEST: ADD {symbol} ({asset_type}) ===")
    try:
        payload = {
            "symbol": symbol,
            "asset_type": asset_type,
            "owns_position": False,
            "notes": f"Test {symbol}",
            "alert_threshold_pct": 5.0,
            "priority": 1
        }
        response = requests.post(
            f"{BASE_URL}/api/v3/watchlist/add",
            json=payload,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        print(json.dumps(data, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_remove_symbol(symbol, asset_type):
    """Test POST /api/v3/watchlist/remove"""
    print(f"\n=== TEST: REMOVE {symbol} ({asset_type}) ===")
    try:
        payload = {
            "symbol": symbol,
            "asset_type": asset_type
        }
        response = requests.post(
            f"{BASE_URL}/api/v3/watchlist/remove",
            json=payload,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        print(json.dumps(data, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("GHOST PROTOCOL - PERSONAL WATCHLIST ENDPOINT TESTS")
    print("=" * 60)
    
    # Test 1: Get empty watchlist
    test_get_watchlist()
    
    # Test 2: Add BTC
    test_add_symbol("BTC", "crypto")
    
    # Test 3: Add AAPL
    test_add_symbol("AAPL", "stock")
    
    # Test 4: Get watchlist with items
    test_get_watchlist()
    
    # Test 5: Remove BTC
    test_remove_symbol("BTC", "crypto")
    
    # Test 6: Get watchlist after removal
    test_get_watchlist()
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
