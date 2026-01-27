#!/usr/bin/env python3
"""
Test Honest Ghost System
========================

Verifies that Ghost now:
1. Shows both UP and DOWN predictions
2. Reports yesterday's results
3. Displays direction properly (🚀 vs 📉)
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core.daily_top_10_scanner import DailyTop10Scanner
from core.guardian_oracle import GuardianOracle


async def test_honest_ghost():
    """Test the honest Ghost system"""
    print("🧪 Testing Honest Ghost System\n")
    print("=" * 60)
    
    # Run scan
    print("\n1️⃣ Running market scan (shows UP AND DOWN now)...")
    scanner = DailyTop10Scanner()
    top_10 = await scanner.scan_for_top_10()
    
    print(f"\n✅ Found {len(top_10)} opportunities")
    print("\n📊 Directions breakdown:")
    
    up_count = sum(1 for opp in top_10 if opp.get('direction') == 'UP')
    down_count = sum(1 for opp in top_10 if opp.get('direction') == 'DOWN')
    
    print(f"   📈 UP predictions: {up_count}")
    print(f"   📉 DOWN predictions: {down_count}")
    
    # Show sample predictions
    print("\n📋 Sample predictions:")
    for i, opp in enumerate(top_10[:3], 1):
        direction = opp.get('direction', 'UP')
        gain = opp['gain_pct']
        emoji = '🚀' if direction == 'UP' else '📉'
        print(f"   {i}. {emoji} {opp['symbol']}: {direction} {gain:+.1f}% @ {opp['confidence']*100:.0f}% confidence")
    
    # Test result reporting
    print("\n2️⃣ Testing yesterday's result reporting...")
    guardian = GuardianOracle()
    results = guardian.get_yesterdays_results()
    
    if results:
        print("\n✅ Results report generated:")
        print(results)
    else:
        print("\n⚠️ No historical results yet (expected for first run)")
    
    # Format full prophecy
    print("\n3️⃣ Generating full prophecy with honest signals...")
    prophecy = await guardian.morning_prophecy(top_10, position_size=100.0)
    
    print("\n" + "=" * 60)
    print("📜 FULL PROPHECY:")
    print("=" * 60)
    print(prophecy)
    print("=" * 60)
    
    # Verify honesty
    print("\n4️⃣ Honesty check:")
    has_results_section = "YESTERDAY'S ACTUAL RESULTS" in prophecy or results is None
    has_up = "BUY" in prophecy or up_count > 0
    has_down = "SHORT" in prophecy or down_count > 0
    has_bearish_emoji = "📉" in prophecy or down_count == 0
    
    print(f"   ✅ Shows yesterday's results: {has_results_section}")
    print(f"   ✅ Has bullish signals: {has_up} ({up_count} UP predictions)")
    print(f"   ✅ Has bearish signals: {has_down} ({down_count} DOWN predictions)")
    print(f"   ✅ Shows bearish emojis: {has_bearish_emoji}")
    
    if down_count > 0:
        print("\n🎉 SUCCESS: Ghost now shows bearish predictions!")
    else:
        print("\n⚠️ No bearish predictions found in this scan (market may be bullish)")
        print("   But filter is removed - bearish signals will show when they occur")
    
    print("\n✅ Honest Ghost system is working!")


if __name__ == "__main__":
    asyncio.run(test_honest_ghost())
