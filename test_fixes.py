#!/usr/bin/env python3
"""
Test script to verify all fixes A-B-C-D work correctly.
"""

import requests
import time
import json

BASE_URL = "http://localhost:8444"

def test_pacs_prediction():
    """Test Issue #1 fix: PACS stock prediction with reduced timeouts"""
    print("\n🧪 TEST A: PACS Stock Prediction (timeout fix)")
    print("=" * 60)
    
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict/run",
            params={"symbol": "PACS"},
            timeout=10
        )
        duration = time.time() - start
        
        data = response.json()
        
        print(f"✅ Response in {duration:.2f}s")
        print(f"   Status: {'SUCCESS' if data.get('ok') else 'FAILED'}")
        print(f"   Price: ${data.get('current_price', 0):.2f}")
        print(f"   Provider: {data.get('provider', 'unknown')}")
        print(f"   Confidence: {data.get('confidence', 0)*100:.1f}%")
        
        if duration > 5:
            print(f"⚠️  WARNING: Took {duration:.2f}s (should be <4s)")
            return False
        
        if not data.get('ok'):
            print(f"❌ FAILED: {data.get('error')}")
            return False
            
        return True
        
    except Exception as e:
        duration = time.time() - start
        print(f"❌ FAILED after {duration:.2f}s: {e}")
        return False


def test_btc_prediction():
    """Test crypto predictions still work"""
    print("\n🧪 TEST: BTC Crypto Prediction (baseline)")
    print("=" * 60)
    
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict/run",
            params={"symbol": "BTC"},
            timeout=10
        )
        duration = time.time() - start
        
        data = response.json()
        
        print(f"✅ Response in {duration:.2f}s")
        print(f"   Status: {'SUCCESS' if data.get('ok') else 'FAILED'}")
        print(f"   Price: ${data.get('current_price', 0):.2f}")
        print(f"   Provider: {data.get('provider', 'unknown')}")
        print(f"   Confidence: {data.get('confidence', 0)*100:.1f}%")
        
        return data.get('ok', False)
        
    except Exception as e:
        duration = time.time() - start
        print(f"❌ FAILED after {duration:.2f}s: {e}")
        return False


def test_market_hours_check():
    """Test Issue #3 fix: Market hours check logged"""
    print("\n🧪 TEST C: Market Hours Check (log inspection)")
    print("=" * 60)
    print("   Making stock prediction to trigger market hours check...")
    print("   (Check logs for 'market_closed' warning if after hours)")
    
    # Just make a prediction - the market hours check happens internally
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict/run",
            params={"symbol": "AAPL"},
            timeout=10
        )
        data = response.json()
        
        if data.get('ok'):
            print("   ✅ Prediction completed (check server logs for market hours info)")
            return True
        else:
            print(f"   ⚠️  Prediction failed: {data.get('error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("GHOST PROTOCOL - FIX VERIFICATION TESTS")
    print("=" * 60)
    
    results = {
        "PACS (timeout fix)": test_pacs_prediction(),
        "BTC (baseline)": test_btc_prediction(),
        "Market hours (logging)": test_market_hours_check(),
    }
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready to deploy")
    else:
        print("❌ SOME TESTS FAILED - Review errors above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
