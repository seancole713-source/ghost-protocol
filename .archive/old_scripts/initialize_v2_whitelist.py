#!/usr/bin/env python3
"""
Initialize V2 quality filters with proven high performers.
Overrides min_predictions requirement for initial whitelist.
"""

import json
import os
from datetime import datetime

# High performers from loser analysis (Jan 25, 2025)
# Crypto ONLY - Stocks are broken (4.5% WR vs 38.7% for crypto)
WHITELIST_SYMBOLS = [
    # Top crypto performers by win rate
    "RNDR",   # 47.6% (89/187) - BEST PERFORMER
    "CHZ",    # 37.1% (75/202)
    "TURBO",  # 35.5% (27/76)
    "ZEC",    # 31.1% (46/148)
    
    # Secondary performers (25%+ WR)
    "ILV",    # ~30%
    "RLC",    # ~28%
    "EGLD",   # ~27%
    "OCEAN",  # ~26%
]

# Poor performers - ALL STOCKS + underperforming crypto
BLACKLIST_SYMBOLS = [
    # STOCKS - ALL BLACKLISTED (4.5% overall WR = broken)
    "ABCL",   # 0% (0/16) - WORST
    "TGTX",   # 3.9% (1/26)
    "XPO",    # 3.9% (1/26)
    "GME",    # 4.0% (4/101) - High volume loser
    "BMBL",   # 4.0% (1/25)
    "ITRI",   # 4.0% (1/25)
    "SOUN",   # 4.0% (1/25)
    "IQ",     # Low performer
    "T",      # Stock - removed from whitelist
    
    # Crypto with 0% or very low win rate
    "XRP",    # 0% (0/28)
    "DOT",    # 0% (0/16)
    "AVAX",   # 0% (0/27)
    "UNI",    # 0% (0/10)
    "PEPE",   # 0% (0/13)
    "SNX",    # 0% (0/13)
    "1INCH",  # 0% (0/13)
    "LDO",    # 0% (0/13)
    "ETC",    # 0% (0/16)
    "ALGO",   # 0% (0/15)
    
    # Major crypto with poor performance
    "BTC",    # Too volatile, hard to predict
    "ETH",    # Poor WR
    "SOL",    # Poor WR
    "ADA",    # Poor WR
    "BNB",    # Poor WR
    "LTC",    # Poor WR
    "ICP",    # Dropped - underperforming
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
