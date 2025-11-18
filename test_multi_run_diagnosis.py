#!/usr/bin/env python3
"""
Diagnostic test for multi-run prediction engine.
Tests price fetching for all configured symbols.
"""
import sys
import time
sys.path.insert(0, '.')

from wolf_app import (
    STOCK_SYMBOLS, CRYPTO_SYMBOLS, VIP_COINS,
    _generate_48h_forecast, _generate_multi_symbol_predictions,
    get_wolf_price, _build_price_providers, _is_market_open_now
)
from core.price_quorum import get_price_quorum

print("═══ GHOST MULTI-RUN DIAGNOSTIC ═══\n")

# Test 1: Verify symbol configuration
print("1️⃣  Symbol Configuration")
print(f"   STOCK_SYMBOLS: {STOCK_SYMBOLS}")
print(f"   CRYPTO_SYMBOLS: {CRYPTO_SYMBOLS[:5]}... (showing first 5)")
print(f"   VIP_COINS: {VIP_COINS}")
print(f"   Total symbols: {len(STOCK_SYMBOLS) + len(CRYPTO_SYMBOLS) + len(VIP_COINS)}")
print()

# Test 2: Test WOLF price (baseline)
print("2️⃣  Baseline WOLF Price Test")
try:
    price, prev, provider = get_wolf_price()
    print(f"   ✅ WOLF: ${price} (provider: {provider})")
except Exception as e:
    print(f"   ❌ WOLF price failed: {e}")
print()

# Test 3: Test providers for key stocks
print("3️⃣  Key Stock Price Providers")
test_stocks = ["AAPL", "MSFT", "NVDA", "GOOGL"]
is_open, _ = _is_market_open_now()
print(f"   Market open: {is_open}")

for symbol in test_stocks:
    print(f"\n   Testing {symbol}:")
    try:
        providers = _build_price_providers(symbol, is_market_open=is_open)
        print(f"     Providers configured: {[p.name for p in providers]}")
        
        decision = get_price_quorum().get_price(
            symbol=symbol,
            providers=providers,
            prev_close=None,
            is_market_open=is_open,
            timeout=30.0
        )
        
        if decision.price:
            print(f"     ✅ Price: ${decision.price} (provider: {decision.provider_label})")
            print(f"     Reason: {decision.reason}, Latency: {decision.latency_ms}ms")
        else:
            print(f"     ❌ Price: None")
            print(f"     Reason: {decision.reason}")
            if decision.quotes:
                print(f"     Provider attempts:")
                for q in decision.quotes:
                    print(f"       - {q.provider}: {q.price or 'FAILED'} ({q.error or 'OK'})")
    except Exception as e:
        print(f"     ❌ Exception: {e}")

# Test 4: Test 48h forecast for one symbol
print("\n\n4️⃣  48h Forecast Test (AAPL)")
try:
    forecast = _generate_48h_forecast("AAPL")
    if forecast.get("ok"):
        print(f"   ✅ Forecast generated")
        print(f"     Current: ${forecast.get('price_now')}")
        print(f"     Predicted (48h): ${forecast.get('price_pred_mid')}")
        print(f"     Confidence: {forecast.get('confidence')}")
    else:
        print(f"   ❌ Forecast failed: {forecast.get('error')}")
except Exception as e:
    print(f"   ❌ Exception: {e}")

# Test 5: Test full multi-run
print("\n\n5️⃣  Full Multi-Symbol Prediction Test")
print("   Running multi-symbol predictions...")
start = time.time()
try:
    result = _generate_multi_symbol_predictions()
    elapsed = time.time() - start
    
    print(f"   Execution time: {elapsed:.1f}s")
    print(f"   Result OK: {result.get('ok')}")
    print(f"   Cached: {result.get('cached', False)}")
    
    counts = result.get('counts', {})
    print(f"\n   Counts:")
    print(f"     Stocks: {counts.get('stocks', 0)}")
    print(f"     Crypto: {counts.get('crypto', 0)}")
    print(f"     VIP: {counts.get('vip', 0)}")
    print(f"     Total: {result.get('total', 0)}")
    
    if result.get('failed_symbols'):
        print(f"\n   Failed symbols:")
        for category, failures in result['failed_symbols'].items():
            if failures:
                print(f"     {category}:")
                for fail in failures[:5]:  # Show first 5
                    print(f"       - {fail.get('symbol')}: {fail.get('error')}")
                if len(failures) > 5:
                    print(f"       ... and {len(failures) - 5} more")
    
    # Show sample predictions
    predictions = result.get('predictions', {})
    for category in ['stocks', 'crypto', 'vip']:
        items = predictions.get(category, [])
        if items:
            print(f"\n   Sample {category} predictions:")
            for pred in items[:3]:  # Show first 3
                symbol = pred.get('symbol')
                price = pred.get('price_current')
                direction = pred.get('direction')
                print(f"     {symbol}: ${price} → {direction}")
            if len(items) > 3:
                print(f"     ... and {len(items) - 3} more")
                
except Exception as e:
    print(f"   ❌ Exception: {e}")
    import traceback
    traceback.print_exc()

print("\n═══ DIAGNOSTIC COMPLETE ═══")
