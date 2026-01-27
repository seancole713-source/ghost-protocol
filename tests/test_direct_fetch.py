#!/usr/bin/env python3
"""Direct test of the updated _get_price_at_time function"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Configure logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Set up environment
os.environ["POLYGON_API_KEY"] = os.getenv("POLYGON_API_KEY", "8VIvELVXiLG30K2l1348RzSurffLM0jR")

# Import fresh (no cache)
sys.path.insert(0, "/workspaces/ghost-protocol")

# Import and reload to get latest code
import services.outcome_reconciler_v2
import importlib
importlib.reload(services.outcome_reconciler_v2)

from services.outcome_reconciler_v2 import _get_price_at_time

print("=" * 70)
print("DIRECT TEST: _get_price_at_time() WITH POLYGON HOURLY BARS")
print("=" * 70)

# Test 48 hours ago (typical Ghost reconciliation window)
symbol = "AAPL"
timestamp = time.time() - (48 * 3600)
dt = datetime.fromtimestamp(timestamp)

print(f"\nTesting: {symbol} @ {dt}")
print(f"Timestamp: {timestamp}")
print()

try:
    price = _get_price_at_time(symbol, timestamp)
    
    if price:
        print(f"\n✅ SUCCESS: ${price:.2f}")
    else:
        print(f"\n❌ FAILED: No price returned")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
