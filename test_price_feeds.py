#!/usr/bin/env python
"""Test which price feeds are working"""
from core.crypto.crypto_providers import get_crypto_price_quorum
from core.providers.turbo_provider import turbo_stock_price
import traceback

# Test BTC (crypto)
print('Testing BTC (crypto)...')
try:
    result = get_crypto_price_quorum('BTC')
    if result and result.get('price'):
        print(f'✅ BTC: ${result["price"]:,.2f} from {result.get("provider", "unknown")}')
    else:
        print(f'❌ BTC: No price data - {result}')
except Exception as e:
    print(f'❌ BTC ERROR: {e}')
    traceback.print_exc()

# Test AAPL (stock)
print('\nTesting AAPL (stock)...')
try:
    result = turbo_stock_price('AAPL', max_budget_s=4.0)
    if result and result.get('price'):
        print(f'✅ AAPL: ${result["price"]:,.2f} from {result.get("provider", "unknown")}')
    else:
        print(f'❌ AAPL: No price data - {result}')
except Exception as e:
    print(f'❌ AAPL ERROR: {e}')
    traceback.print_exc()

# Test ETH (crypto)
print('\nTesting ETH (crypto)...')
try:
    result = get_crypto_price_quorum('ETH')
    if result and result.get('price'):
        print(f'✅ ETH: ${result["price"]:,.2f} from {result.get("provider", "unknown")}')
    else:
        print(f'❌ ETH: No price data - {result}')
except Exception as e:
    print(f'❌ ETH ERROR: {e}')
    traceback.print_exc()
