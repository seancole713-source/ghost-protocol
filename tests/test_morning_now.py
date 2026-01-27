#!/usr/bin/env python3
"""
Send Morning Prophecy RIGHT NOW to Telegram
===========================================

This triggers an immediate morning prophecy (doesn't wait for 6 AM).
"""

import asyncio
import sys
sys.path.insert(0, '.')

async def send_morning_now():
    """Send morning prophecy to Telegram immediately"""
    
    from core.guardian_heartbeat_scheduler import GuardianHeartbeatScheduler
    
    print("=" * 70)
    print("📱 SENDING MORNING PROPHECY TO TELEGRAM NOW")
    print("=" * 70)
    print()
    
    # Create scheduler
    scheduler = GuardianHeartbeatScheduler()
    
    print("🎬 Using DEMO MODE (forced positive predictions)")
    print("📡 Scanning market...")
    print()
    
    # Trigger morning heartbeat with $100 position sizing
    await scheduler._send_heartbeat('morning', position_size=100.0)
    
    print()
    print("=" * 70)
    print("✅ SENT!")
    print("=" * 70)
    print()
    print("📱 Check your Telegram now!")
    print()

if __name__ == "__main__":
    asyncio.run(send_morning_now())
