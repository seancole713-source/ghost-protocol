#!/usr/bin/env python3
"""
Initialize Ghost Protocol Watchlist
Populates watchlist with default symbols for predictions
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def init_watchlist():
    """Populate watchlist with default symbols"""
    try:
        from core.watchlist_manager import WatchlistManager
        
        wm = WatchlistManager()
        
        print("📋 Initializing Ghost Protocol Watchlist...")
        
        # VIP symbols (always track)
        vip_symbols = [
            ("WOLF", "Wolfspeed Inc.")
        ]
        
        # Top stocks by market cap
        stock_symbols = [
            ("AAPL", "Apple Inc."),
            ("MSFT", "Microsoft Corporation"),
            ("NVDA", "NVIDIA Corporation"),
            ("GOOGL", "Alphabet Inc."),
            ("AMZN", "Amazon.com Inc."),
            ("META", "Meta Platforms Inc."),
            ("TSLA", "Tesla Inc."),
            ("AMD", "Advanced Micro Devices"),
            ("NFLX", "Netflix Inc."),
            ("DIS", "Walt Disney Company"),
            ("BA", "Boeing Company"),
            ("JPM", "JPMorgan Chase"),
            ("V", "Visa Inc."),
            ("MA", "Mastercard Inc."),
            ("HD", "Home Depot"),
            ("NKE", "Nike Inc."),
            ("MCD", "McDonald's Corporation"),
            ("SBUX", "Starbucks Corporation"),
            ("COST", "Costco Wholesale"),
            ("WMT", "Walmart Inc.")
        ]
        
        # Top cryptocurrencies by market cap
        crypto_symbols = [
            ("BTC", "Bitcoin"),
            ("ETH", "Ethereum"),
            ("SOL", "Solana"),
            ("BNB", "Binance Coin"),
            ("XRP", "Ripple"),
            ("ADA", "Cardano"),
            ("AVAX", "Avalanche"),
            ("DOT", "Polkadot"),
            ("MATIC", "Polygon"),
            ("LINK", "Chainlink")
        ]
        
        # Add VIP symbols
        print("\n🌟 Adding VIP symbols...")
        for symbol, name in vip_symbols:
            result = wm.add_symbol(symbol, name=name)
            print(f"  ✓ {symbol} - {name}")
        
        # Add stock symbols
        print("\n📈 Adding stock symbols...")
        for symbol, name in stock_symbols:
            result = wm.add_symbol(symbol, name=name)
            print(f"  ✓ {symbol} - {name}")
        
        # Add crypto symbols
        print("\n₿ Adding crypto symbols...")
        for symbol, name in crypto_symbols:
            result = wm.add_symbol(symbol, name=name)
            if result.get("success"):
                print(f"  ✓ {symbol} - {name}")
            else:
                print(f"  ⚠️  {symbol} - {result.get('error', 'Unknown error')}")
        
        # Verify
        watchlist = wm.get_watchlist()
        total = len(watchlist)
        
        print(f"\n✅ Watchlist initialized: {total} symbols")
        print(f"   VIP: {len(vip_symbols)}")
        print(f"   Stocks: {len(stock_symbols)}")
        print(f"   Crypto: {len(crypto_symbols)}")
        
        return total
        
    except Exception as e:
        print(f"❌ Error initializing watchlist: {e}")
        import traceback
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    count = init_watchlist()
    sys.exit(0 if count > 0 else 1)
