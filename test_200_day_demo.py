#!/usr/bin/env python3
"""
Test $200/Day Passive Income Machine - Demo Mode
================================================

This script demonstrates the morning prophecy format with forced positive predictions.

Run:
    python3 test_200_day_demo.py
"""

import asyncio
import sys
sys.path.insert(0, '.')

async def demo_200_day_machine():
    """Demo the $200/day format with forced positive predictions"""
    
    from core.daily_top_10_scanner import DailyTop10Scanner
    from core.guardian_oracle import get_guardian_oracle
    
    print("=" * 70)
    print("🎬 DEMO: $200/DAY PASSIVE INCOME MACHINE")
    print("=" * 70)
    print()
    
    # Enable demo mode: Force positive predictions
    scanner = DailyTop10Scanner(demo_mode=True)
    
    print("📡 Scanning market for top 10 opportunities...")
    print("   (Demo mode: Forcing positive predictions)")
    print()
    
    # Scan for top 10
    top_10 = await scanner.scan_for_top_10()
    
    print(f"✅ Found {len(top_10)} opportunities\n")
    
    # Show brief summary
    total_gain = sum(opp['gain_pct'] for opp in top_10)
    avg_gain = total_gain / len(top_10) if top_10 else 0
    avg_confidence = sum(opp['confidence'] for opp in top_10) / len(top_10) if top_10 else 0
    
    print(f"📊 Summary:")
    print(f"   Total potential gain: {total_gain:.1f}%")
    print(f"   Average gain per position: {avg_gain:.1f}%")
    print(f"   Average confidence: {avg_confidence:.1%}")
    print()
    
    # Generate morning prophecy with $100 position sizing
    guardian = get_guardian_oracle()
    
    print("=" * 70)
    print("💰 YOUR MORNING PROPHECY ($100 POSITION SIZING)")
    print("=" * 70)
    print()
    
    message = await guardian.morning_prophecy(top_10, position_size=100.0)
    
    print(message)
    
    print()
    print("=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)
    print()
    print("📱 To enable demo mode in production:")
    print("   Railway: Set env var GHOST_DEMO_MODE=1")
    print("   Local: export GHOST_DEMO_MODE=1")
    print()
    print("💡 To use REAL ML predictions:")
    print("   Railway: Remove GHOST_DEMO_MODE env var")
    print("   Local: unset GHOST_DEMO_MODE")
    print()

if __name__ == "__main__":
    asyncio.run(demo_200_day_machine())
