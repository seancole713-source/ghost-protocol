#!/usr/bin/env python3
"""
Initialize Smart Watcher with essential watchlist (25 ticker limit)
Priority: VIP (1) + Top stocks (14) + Top crypto (10) = 25 total
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.smart_watcher import get_smart_watcher

def main():
    print("📋 Initializing Smart Watcher (25 ticker limit)...")
    
    # Priority symbols (25 total)
    symbols = [
        # VIP (1)
        "WOLF",
        
        # Top Stocks (14)
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
        "AMD", "NFLX", "DIS", "BA", "JPM", "V", "MA",
        
        # Top Crypto (10)
        "BTC", "ETH", "SOL", "BNB", "XRP",
        "ADA", "AVAX", "DOT", "MATIC", "LINK"
    ]
    
    try:
        watcher = get_smart_watcher()
        
        # Get existing watchlist
        existing = watcher.get_watchlist()
        existing_symbols = {t.symbol if hasattr(t, 'symbol') else t.get('symbol', '') 
                           for t in existing}
        
        print(f"\n📊 Current watchlist: {len(existing_symbols)} symbols")
        
        # Add new symbols (up to capacity)
        added = 0
        skipped = 0
        failed = 0
        
        for symbol in symbols:
            if symbol in existing_symbols:
                skipped += 1
                continue
                
            try:
                result = watcher.add_ticker(symbol)
                if result.get("success"):
                    print(f"  ✓ {symbol}")
                    added += 1
                else:
                    msg = result.get("message", "Unknown error")
                    if "full" in msg.lower():
                        print(f"  ⚠️  Capacity reached at {added + len(existing_symbols)} symbols")
                        break
                    else:
                        print(f"  ⚠️  {symbol} - {msg}")
                        failed += 1
            except Exception as e:
                print(f"  ❌ Error adding {symbol}: {e}")
                failed += 1
        
        # Verify final state
        final = watcher.get_watchlist()
        print(f"\n✅ Smart Watcher initialized")
        print(f"   Added: {added}")
        print(f"   Skipped (already exists): {skipped}")
        print(f"   Failed: {failed}")
        print(f"   Total symbols: {len(final)}/25")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
