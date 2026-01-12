#!/usr/bin/env python3
"""
Initialize V2 quality filters with proven high performers.
Overrides min_predictions requirement for initial whitelist.
"""

import json
import os
from datetime import datetime

# High performers from 30-day analysis (90%+ win rate)
WHITELIST_SYMBOLS = [
    "RLC",    # 100% (5/5)
    "EGLD",   # 100% (5/5)
    "RNDR",   # 100% (12/12)
    "ZEC",    # 100% (7/7)
    "ILV",    # 100% (13/13)
    "T",      # 100% (18/18)
    "TURBO",  # 100% (13/13)
    "CHZ",    # 100% (13/13)
    "ICP",    # 93.3% (14/15)
    "OCEAN",  # 90.0% (9/10)
]

# Poor performers (0% win rate or < 45%)
BLACKLIST_SYMBOLS = [
    # Crypto with 0% win rate
    "XRP",    # 0% (0/28)
    "DOT",    # 0% (0/16)
    "AVAX",   # 0% (0/27)
    
    # Stocks with 0% win rate
    "UNI",    # 0% (0/10)
    "PEPE",   # 0% (0/13)
    "SNX",    # 0% (0/13)
    "1INCH",  # 0% (0/13)
    "LDO",    # 0% (0/13)
    "ETC",    # 0% (0/16)
    "ALGO",   # 0% (0/15)
    
    # Major crypto with poor performance (from API)
    "BTC",
    "ETH",
    "SOL",
    "ADA",
    "BNB",
    "LTC",
]

def initialize_filters():
    """Create initial V2 quality filter configuration."""
    
    config = {
        "whitelist": sorted(list(set(WHITELIST_SYMBOLS))),
        "blacklist": sorted(list(set(BLACKLIST_SYMBOLS))),
        "metrics": {},
        "last_updated": datetime.utcnow().isoformat(),
        "config": {
            "min_predictions": 20,
            "whitelist_wr": 0.55,
            "blacklist_wr": 0.45,
            "note": "Initial whitelist manually set for proven 90%+ performers"
        }
    }
    
    # Save to ghost_v2_quality.json
    with open("ghost_v2_quality.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("=" * 80)
    print("V2 QUALITY FILTERS INITIALIZED")
    print("=" * 80)
    print(f"✅ Whitelist: {len(config['whitelist'])} symbols")
    for symbol in config['whitelist']:
        print(f"   ✅ {symbol}")
    print()
    print(f"❌ Blacklist: {len(config['blacklist'])} symbols")
    for symbol in config['blacklist']:
        print(f"   ❌ {symbol}")
    print("=" * 80)
    print(f"Config saved to: ghost_v2_quality.json")
    print()
    print("NEXT STEPS:")
    print("1. Restart Wolf app to load new filters")
    print("2. Integrate V2 filter into daily TOP 10 flow")
    print("3. Only predict whitelisted symbols from now on")

if __name__ == "__main__":
    initialize_filters()
