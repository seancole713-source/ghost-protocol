#!/usr/bin/env python3
"""
Simulate V2 filter on sample predictions to show strict mode in action.
"""

from core.v2_quality import get_quality_system

# Simulate a typical prediction batch (what Ghost might generate)
SAMPLE_PREDICTIONS = [
    ("AAPL", 0.85, "stock"),
    ("BTC", 0.90, "crypto"),
    ("RLC", 0.78, "stock"),     # ✅ Whitelisted
    ("TSLA", 0.82, "stock"),
    ("ETH", 0.88, "crypto"),
    ("RNDR", 0.76, "stock"),    # ✅ Whitelisted
    ("MSFT", 0.84, "stock"),
    ("XRP", 0.87, "crypto"),
    ("ICP", 0.79, "stock"),     # ✅ Whitelisted
    ("NVDA", 0.91, "stock"),
    ("SOL", 0.89, "crypto"),
    ("CHZ", 0.74, "stock"),     # ✅ Whitelisted
    ("META", 0.86, "stock"),
    ("PEPE", 0.83, "crypto"),
    ("OCEAN", 0.77, "stock"),   # ✅ Whitelisted
    ("LRC", 0.80, "crypto"),    # ❌ NOT whitelisted (typo confusion)
    ("AVAX", 0.85, "crypto"),
    ("ZEC", 0.75, "stock"),     # ✅ Whitelisted
    ("GOOGL", 0.88, "stock"),
    ("DOT", 0.86, "crypto"),
]

def test_v2_filter():
    quality = get_quality_system()
    
    print("=" * 80)
    print("V2 STRICT MODE SIMULATION - NEXT TOP 10 PREDICTIONS")
    print("=" * 80)
    print()
    print("INPUT: 20 predictions from various sources")
    print("OUTPUT: Only whitelisted symbols allowed through")
    print()
    print("-" * 80)
    
    allowed = []
    blocked = []
    
    for symbol, confidence, asset_type in SAMPLE_PREDICTIONS:
        should_predict, reason = quality.should_predict(symbol, confidence)
        
        if should_predict:
            allowed.append((symbol, confidence, reason))
        else:
            blocked.append((symbol, confidence, reason))
    
    # Show allowed (would be in TOP 10)
    print(f"\n✅ ALLOWED ({len(allowed)} symbols):")
    print(f"{'Symbol':<10} {'Conf':<8} {'Reason':<50}")
    print("-" * 80)
    for symbol, conf, reason in sorted(allowed, key=lambda x: x[1], reverse=True):
        print(f"{symbol:<10} {conf*100:.0f}%     {reason}")
    
    # Show blocked
    print(f"\n❌ BLOCKED ({len(blocked)} symbols):")
    print(f"{'Symbol':<10} {'Conf':<8} {'Reason':<50}")
    print("-" * 80)
    for symbol, conf, reason in sorted(blocked, key=lambda x: x[1], reverse=True)[:10]:
        print(f"{symbol:<10} {conf*100:.0f}%     {reason}")
    if len(blocked) > 10:
        print(f"... and {len(blocked) - 10} more")
    
    print()
    print("=" * 80)
    print("SUMMARY:")
    print(f"  • Started with: {len(SAMPLE_PREDICTIONS)} predictions")
    print(f"  • Allowed: {len(allowed)} symbols (whitelist only)")
    print(f"  • Blocked: {len(blocked)} symbols")
    print(f"  • Filter effectiveness: {len(blocked)/len(SAMPLE_PREDICTIONS)*100:.0f}% filtered out")
    print()
    print("🎯 V2 STRICT MODE: Only proven 90%+ win rate symbols allowed")
    print("=" * 80)

if __name__ == "__main__":
    test_v2_filter()
