#!/usr/bin/env python3
"""
Test yfinance directly to diagnose SPY/VIX issues
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 80)
print("TESTING YFINANCE INSTALLATION AND DATA FETCH")
print("=" * 80)

# Test 1: Can we import yfinance?
print("\n1. Testing yfinance import...")
try:
    import yfinance as yf
    print("   ✅ yfinance imported successfully")
except ImportError as e:
    print(f"   ❌ Cannot import yfinance: {e}")
    print("   💡 Install with: pip install yfinance")
    sys.exit(1)

# Test 2: Can we fetch SPY?
print("\n2. Testing SPY data fetch...")
try:
    spy = yf.Ticker("SPY")
    spy_data = spy.history(period="2d")
    
    if not spy_data.empty:
        current_price = float(spy_data['Close'].iloc[-1])
        prev_close = float(spy_data['Close'].iloc[-2]) if len(spy_data) >= 2 else current_price
        change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
        
        print(f"   ✅ SPY: ${current_price:.2f} ({change_pct:+.2f}%)")
        print(f"   📊 Data points: {len(spy_data)}")
    else:
        print("   ❌ SPY: No data returned (empty DataFrame)")
except Exception as e:
    print(f"   ❌ SPY error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Can we fetch ^GSPC?
print("\n3. Testing ^GSPC data fetch...")
try:
    gspc = yf.Ticker("^GSPC")
    gspc_data = gspc.history(period="2d")
    
    if not gspc_data.empty:
        current_price = float(gspc_data['Close'].iloc[-1])
        prev_close = float(gspc_data['Close'].iloc[-2]) if len(gspc_data) >= 2 else current_price
        change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
        
        print(f"   ✅ ^GSPC: ${current_price:.2f} ({change_pct:+.2f}%)")
        print(f"   📊 Data points: {len(gspc_data)}")
    else:
        print("   ❌ ^GSPC: No data returned (empty DataFrame)")
except Exception as e:
    print(f"   ❌ ^GSPC error: {e}")

# Test 4: Can we fetch ^VIX?
print("\n4. Testing ^VIX data fetch...")
try:
    vix = yf.Ticker("^VIX")
    vix_data = vix.history(period="2d")
    
    if not vix_data.empty:
        vix_level = float(vix_data['Close'].iloc[-1])
        prev_close = float(vix_data['Close'].iloc[-2]) if len(vix_data) >= 2 else vix_level
        change = vix_level - prev_close
        
        print(f"   ✅ ^VIX: {vix_level:.2f} (change: {change:+.2f})")
        print(f"   📊 Data points: {len(vix_data)}")
    else:
        print("   ❌ ^VIX: No data returned (empty DataFrame)")
except Exception as e:
    print(f"   ❌ ^VIX error: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("If all tests passed, yfinance is working.")
print("If tests failed, check:")
print("  - Is yfinance installed? (pip list | grep yfinance)")
print("  - Is internet accessible from Railway?")
print("  - Is Yahoo Finance API rate-limiting?")
