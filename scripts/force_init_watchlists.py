#!/usr/bin/env python3
"""
FORCE initialize watchlists on Railway
Run this via Railway console: python3 scripts/force_init_watchlists.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def init_smart_watcher():
    """Initialize Smart Watcher with 25 symbols"""
    try:
        from core.smart_watcher import get_smart_watcher
        
        symbols = [
            "WOLF",  # VIP
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",  # Top stocks
            "AMD", "NFLX", "DIS", "BA", "JPM", "V", "MA",
            "BTC", "ETH", "SOL", "BNB", "XRP",  # Top crypto
            "ADA", "AVAX", "DOT", "MATIC", "LINK"
        ]
        
        watcher = get_smart_watcher()
        existing = watcher.get_watchlist()
        existing_symbols = {t.symbol if hasattr(t, 'symbol') else t.get('symbol', '') 
                           for t in existing}
        
        print(f"Smart Watcher: {len(existing_symbols)} existing symbols")
        
        added = 0
        for symbol in symbols:
            if symbol in existing_symbols:
                continue
            result = watcher.add_ticker(symbol)
            if result.get("success"):
                added += 1
                print(f"  ✓ {symbol}")
            elif "full" in result.get("message", "").lower():
                print(f"  ⚠️  Capacity reached")
                break
                
        final = watcher.get_watchlist()
        print(f"✅ Smart Watcher: {len(final)}/25 symbols ({added} added)")
        return True
    except Exception as e:
        print(f"❌ Smart Watcher failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def init_watchlist_manager():
    """Initialize Watchlist Manager with 82 symbols"""
    try:
        from core.watchlist_manager import WatchlistManager
        
        symbols = [
            # VIP
            ("WOLF", "Wolfspeed Inc."),
            # Stocks (20)
            ("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corporation"),
            ("NVDA", "NVIDIA Corporation"), ("GOOGL", "Alphabet Inc."),
            ("AMZN", "Amazon.com Inc."), ("META", "Meta Platforms Inc."),
            ("TSLA", "Tesla Inc."), ("AMD", "Advanced Micro Devices"),
            ("NFLX", "Netflix Inc."), ("DIS", "Walt Disney Company"),
            ("BA", "Boeing Company"), ("JPM", "JPMorgan Chase"),
            ("V", "Visa Inc."), ("MA", "Mastercard Inc."),
            ("HD", "Home Depot"), ("NKE", "Nike Inc."),
            ("MCD", "McDonald's Corporation"), ("SBUX", "Starbucks Corporation"),
            ("COST", "Costco Wholesale"), ("WMT", "Walmart Inc."),
            # Crypto (10)
            ("BTC", "Bitcoin"), ("ETH", "Ethereum"), ("SOL", "Solana"),
            ("BNB", "Binance Coin"), ("XRP", "Ripple"), ("ADA", "Cardano"),
            ("AVAX", "Avalanche"), ("DOT", "Polkadot"),
            ("MATIC", "Polygon"), ("LINK", "Chainlink"),
        ]
        
        wm = WatchlistManager()
        existing = wm.get_watchlist()
        existing_symbols = {s['symbol'] for s in existing}
        
        print(f"Watchlist Manager: {len(existing_symbols)} existing symbols")
        
        added = 0
        for symbol, name in symbols:
            if symbol in existing_symbols:
                continue
            result = wm.add_symbol(symbol, name=name)
            if result.get("success"):
                added += 1
                print(f"  ✓ {symbol}")
                
        final = wm.get_watchlist()
        print(f"✅ Watchlist Manager: {len(final)} symbols ({added} added)")
        return True
    except Exception as e:
        print(f"❌ Watchlist Manager failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔥 FORCE initializing watchlists...")
    print("")
    
    sw_ok = init_smart_watcher()
    print("")
    wm_ok = init_watchlist_manager()
    print("")
    
    if sw_ok and wm_ok:
        print("✅ All watchlists initialized successfully!")
        sys.exit(0)
    else:
        print("⚠️  Some watchlists failed to initialize")
        sys.exit(1)
