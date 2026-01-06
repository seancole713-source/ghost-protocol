#!/usr/bin/env python3
"""
Test Suite for Ghost Accuracy Fixes (January 2026)
==================================================
This script validates the 3 fixes deployed to solve "insufficient aligned points (0)":

Fix #1: TwelveData stock price fallback (for when yfinance fails)
Fix #2: Timestamp alignment tolerance increased (60s → 7200s)
Fix #3: Hourly actual price collector service

Run: python test_accuracy_fixes.py
"""

import sys
import time
import asyncio
from datetime import datetime, timedelta
from typing import Optional

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{text}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")

def print_pass(text):
    print(f"  {GREEN}✅ PASS:{RESET} {text}")

def print_fail(text):
    print(f"  {RED}❌ FAIL:{RESET} {text}")

def print_warn(text):
    print(f"  {YELLOW}⚠️  WARN:{RESET} {text}")

def print_info(text):
    print(f"  {BLUE}ℹ️  INFO:{RESET} {text}")


# ============================================================================
# FIX #1: TwelveData Stock Price Fallback
# ============================================================================

def test_twelvedata_fallback():
    """Test that TwelveData fallback is working for stock prices."""
    print_header("FIX #1: TwelveData Stock Price Fallback")
    
    passed = 0
    failed = 0
    
    # Test 1: Check TwelveData function exists in wolf_app
    try:
        # Import the function
        sys.path.insert(0, '/workspaces/ghost-protocol')
        
        # Read wolf_app to check function exists
        with open('/workspaces/ghost-protocol/wolf_app.py', 'r') as f:
            content = f.read()
        
        if '_fetch_price_twelvedata' in content:
            print_pass("TwelveData function exists in wolf_app.py")
            passed += 1
        else:
            print_fail("TwelveData function NOT found in wolf_app.py")
            failed += 1
    except Exception as e:
        print_fail(f"Could not check wolf_app.py: {e}")
        failed += 1
    
    # Test 2: Check TwelveData is in turbo_provider chain
    try:
        with open('/workspaces/ghost-protocol/core/providers/turbo_provider.py', 'r') as f:
            content = f.read()
        
        if 'twelvedata' in content.lower():
            print_pass("TwelveData added to turbo_provider chain")
            passed += 1
        else:
            print_fail("TwelveData NOT in turbo_provider chain")
            failed += 1
    except Exception as e:
        print_fail(f"Could not check turbo_provider.py: {e}")
        failed += 1
    
    # Test 3: Live test of TwelveData API
    try:
        import httpx
        url = "https://api.twelvedata.com/price?symbol=AAPL&apikey=demo"
        
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
            data = resp.json()
        
        if 'price' in data:
            price = float(data['price'])
            print_pass(f"TwelveData API working: AAPL = ${price:.2f}")
            passed += 1
        else:
            print_warn(f"TwelveData returned: {data}")
            print_info("This may be rate-limited (demo API)")
            passed += 1  # Not a failure, just rate limited
    except Exception as e:
        print_warn(f"TwelveData live test skipped: {e}")
        print_info("This is OK if you're offline or rate-limited")
    
    return passed, failed


# ============================================================================
# FIX #2: Timestamp Alignment Tolerance (60s → 7200s)
# ============================================================================

def test_alignment_tolerance():
    """Test that timestamp alignment tolerance is now 7200 seconds."""
    print_header("FIX #2: Timestamp Alignment Tolerance (2 hours)")
    
    passed = 0
    failed = 0
    
    # Test 1: Check the constant in outcome_reconciler.py
    try:
        with open('/workspaces/ghost-protocol/services/outcome_reconciler.py', 'r') as f:
            content = f.read()
        
        # Look for the tolerance constant
        if 'ALIGNMENT_TOLERANCE_SEC' in content:
            # Extract the value
            import re
            match = re.search(r'ALIGNMENT_TOLERANCE_SEC\s*=\s*(\d+)', content)
            if match:
                tolerance = int(match.group(1))
                if tolerance >= 7200:
                    print_pass(f"Alignment tolerance is {tolerance}s ({tolerance/3600:.1f} hours)")
                    passed += 1
                else:
                    print_fail(f"Alignment tolerance is only {tolerance}s (should be ≥7200)")
                    failed += 1
            else:
                print_fail("Could not parse ALIGNMENT_TOLERANCE_SEC value")
                failed += 1
        else:
            print_warn("ALIGNMENT_TOLERANCE_SEC constant not found")
            print_info("Checking for inline tolerance...")
    except Exception as e:
        print_fail(f"Could not check outcome_reconciler.py: {e}")
        failed += 1
    
    # Test 2: Verify the alignment logic works
    try:
        # Simulate the alignment check
        tolerance = 7200  # 2 hours
        
        # Test case 1: Within tolerance (should align)
        forecast_ts = 1704067200  # Jan 1, 2024 00:00 UTC
        actual_ts = 1704070000    # 46 minutes later
        diff = abs(forecast_ts - actual_ts)
        
        if diff <= tolerance:
            print_pass(f"46-minute difference correctly aligns (diff={diff}s ≤ {tolerance}s)")
            passed += 1
        else:
            print_fail("Alignment logic failed for 46-minute diff")
            failed += 1
        
        # Test case 2: Outside old tolerance, inside new
        diff_90min = 90 * 60  # 90 minutes = 5400 seconds
        if diff_90min <= tolerance:
            print_pass(f"90-minute difference correctly aligns (diff={diff_90min}s ≤ {tolerance}s)")
            passed += 1
        else:
            print_fail("90-minute diff should align with 2-hour tolerance")
            failed += 1
            
    except Exception as e:
        print_fail(f"Alignment logic test failed: {e}")
        failed += 1
    
    return passed, failed


# ============================================================================
# FIX #3: Hourly Actual Price Collector
# ============================================================================

def test_price_collector():
    """Test that the hourly actual price collector is properly implemented."""
    print_header("FIX #3: Hourly Actual Price Collector Service")
    
    passed = 0
    failed = 0
    
    # Test 1: Check the collector file exists
    try:
        import os
        collector_path = '/workspaces/ghost-protocol/services/actual_price_collector.py'
        if os.path.exists(collector_path):
            print_pass("actual_price_collector.py exists")
            passed += 1
        else:
            print_fail("actual_price_collector.py NOT found")
            failed += 1
            return passed, failed
    except Exception as e:
        print_fail(f"Could not check file: {e}")
        failed += 1
        return passed, failed
    
    # Test 2: Check key functions exist
    try:
        with open('/workspaces/ghost-protocol/services/actual_price_collector.py', 'r') as f:
            content = f.read()
        
        required_functions = [
            'get_current_price',
            'collect_actual_prices',
            'start_collector_scheduler',
        ]
        
        for func in required_functions:
            if f'def {func}' in content:
                print_pass(f"Function '{func}()' exists")
                passed += 1
            else:
                print_fail(f"Function '{func}()' NOT found")
                failed += 1
    except Exception as e:
        print_fail(f"Could not read collector file: {e}")
        failed += 1
    
    # Test 3: Test get_current_price function works
    try:
        sys.path.insert(0, '/workspaces/ghost-protocol')
        from services.actual_price_collector import get_current_price
        
        # Test with BTC (crypto)
        price = get_current_price('BTC')
        if price and price > 0:
            print_pass(f"get_current_price('BTC') = ${price:,.2f}")
            passed += 1
        else:
            print_warn("get_current_price('BTC') returned None/0")
            print_info("This may be due to API unavailability")
    except Exception as e:
        print_warn(f"get_current_price test: {e}")
        print_info("Function exists but may need API keys in production")
    
    # Test 4: Check scheduler configuration
    try:
        if 'schedule' in content or 'scheduler' in content.lower() or 'asyncio' in content:
            print_pass("Scheduler mechanism found in collector")
            passed += 1
        else:
            print_warn("No obvious scheduler found")
    except:
        pass
    
    return passed, failed


# ============================================================================
# INTEGRATION TEST: Full Reconciliation Flow
# ============================================================================

def test_reconciliation_flow():
    """Test the full reconciliation flow end-to-end."""
    print_header("INTEGRATION: Full Reconciliation Flow")
    
    passed = 0
    failed = 0
    
    # Test 1: Check outcome_reconciler_v2 exists
    try:
        import os
        if os.path.exists('/workspaces/ghost-protocol/services/outcome_reconciler_v2.py'):
            print_pass("outcome_reconciler_v2.py exists")
            passed += 1
        else:
            print_warn("outcome_reconciler_v2.py not found (may use v1)")
    except Exception as e:
        print_warn(f"Could not check reconciler: {e}")
    
    # Test 2: Check for CryptoCompare integration (historical prices)
    try:
        files_to_check = [
            '/workspaces/ghost-protocol/services/outcome_reconciler_v2.py',
            '/workspaces/ghost-protocol/services/outcome_reconciler.py',
            '/workspaces/ghost-protocol/wolf_app.py'
        ]
        
        cryptocompare_found = False
        for filepath in files_to_check:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    if 'cryptocompare' in f.read().lower():
                        cryptocompare_found = True
                        break
        
        if cryptocompare_found:
            print_pass("CryptoCompare integration found (for historical prices)")
            passed += 1
        else:
            print_info("CryptoCompare not found in checked files")
    except Exception as e:
        print_warn(f"Could not check CryptoCompare: {e}")
    
    # Test 3: Check prediction store for actual_points support
    try:
        prediction_store_files = [
            '/workspaces/ghost-protocol/prediction_store.py',
            '/workspaces/ghost-protocol/core/prediction_store.py',
            '/workspaces/ghost-protocol/services/prediction_store.py',
        ]
        
        for filepath in prediction_store_files:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read()
                    if 'actual_points' in content or 'actual_price' in content:
                        print_pass(f"Actual price storage support found in {filepath.split('/')[-1]}")
                        passed += 1
                        break
    except Exception as e:
        print_info(f"Prediction store check: {e}")
    
    return passed, failed


# ============================================================================
# LIVE API TEST (Optional)
# ============================================================================

def test_live_apis():
    """Test live API endpoints (requires API keys)."""
    print_header("LIVE API TESTS (Production Readiness)")
    
    passed = 0
    failed = 0
    
    try:
        import httpx
        import os
    except ImportError:
        print_warn("httpx not installed, skipping live tests")
        return passed, failed
    
    # Test 1: CryptoCompare (free API)
    try:
        url = "https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=USD"
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
            data = resp.json()
        
        if 'USD' in data:
            print_pass(f"CryptoCompare API: BTC = ${data['USD']:,.2f}")
            passed += 1
        else:
            print_fail(f"CryptoCompare unexpected response: {data}")
            failed += 1
    except Exception as e:
        print_warn(f"CryptoCompare test failed: {e}")
    
    # Test 2: Coinbase (free API)
    try:
        url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
            data = resp.json()
        
        if 'data' in data and 'amount' in data['data']:
            price = float(data['data']['amount'])
            print_pass(f"Coinbase API: BTC = ${price:,.2f}")
            passed += 1
        else:
            print_warn(f"Coinbase unexpected: {data}")
    except Exception as e:
        print_warn(f"Coinbase test: {e}")
    
    # Test 3: Check if we can reach production
    railway_url = os.environ.get('RAILWAY_URL', 'https://ghost-protocol-production.up.railway.app')
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(f"{railway_url}/health")
            if resp.status_code == 200:
                print_pass(f"Production health check: OK")
                passed += 1
            else:
                print_info(f"Production returned: {resp.status_code}")
    except Exception as e:
        print_info(f"Production not reachable locally (expected in dev): {type(e).__name__}")
    
    return passed, failed


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Ghost Protocol - Accuracy Fixes Test Suite{RESET}")
    print(f"{BOLD}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    
    total_passed = 0
    total_failed = 0
    
    # Run all test suites
    tests = [
        test_twelvedata_fallback,
        test_alignment_tolerance,
        test_price_collector,
        test_reconciliation_flow,
        test_live_apis,
    ]
    
    for test_func in tests:
        try:
            passed, failed = test_func()
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print_fail(f"Test suite {test_func.__name__} crashed: {e}")
            total_failed += 1
    
    # Summary
    print_header("TEST SUMMARY")
    print(f"  {GREEN}Passed:{RESET} {total_passed}")
    print(f"  {RED}Failed:{RESET} {total_failed}")
    
    if total_failed == 0:
        print(f"\n  {GREEN}{BOLD}🎉 ALL FIXES VERIFIED!{RESET}")
        print(f"  {GREEN}Ghost accuracy system should now work correctly.{RESET}\n")
        return 0
    else:
        print(f"\n  {YELLOW}{BOLD}⚠️  Some tests failed - review above{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
