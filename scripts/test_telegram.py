"""
Telegram Integration Test Script (Phase 5.2)

Tests Telegram alert functionality end-to-end.
Sends test messages to verify configuration is correct.

Usage:
    python scripts/test_telegram.py

Ghost Protocol v5 — Session 6
"""

import os
import sys
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_telegram_alerts():
    """Test Telegram integration with various message types."""
    print("🧪 Testing Telegram Integration...")
    print("=" * 60)
    
    # Check environment variables
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not set in environment")
        print("   Set it in Railway dashboard or .env file")
        return False
    
    if not chat_id:
        print("❌ TELEGRAM_CHAT_ID not set in environment")
        print("   Set it in Railway dashboard or .env file")
        return False
    
    print(f"✅ TELEGRAM_BOT_TOKEN: {bot_token[:10]}...{bot_token[-5:]}")
    print(f"✅ TELEGRAM_CHAT_ID: {chat_id}")
    print()
    
    # Try to import notification system
    try:
        from core.notification_system import NotificationSystem
        print("✅ Imported NotificationSystem")
    except ImportError as e:
        print(f"❌ Failed to import NotificationSystem: {e}")
        return False
    
    # Initialize notification system
    try:
        notif = NotificationSystem()
        print("✅ Initialized NotificationSystem")
    except Exception as e:
        print(f"❌ Failed to initialize NotificationSystem: {e}")
        return False
    
    print()
    print("📤 Sending test messages...")
    print("-" * 60)
    
    # Test 1: Simple text message
    print("\n[Test 1] Sending simple text message...")
    try:
        result = await notif.send_message("🧪 Ghost Protocol Test Message\n\nIf you see this, Telegram integration is working!")
        if result:
            print("✅ Simple message sent successfully")
        else:
            print("❌ Simple message failed (returned False)")
    except Exception as e:
        print(f"❌ Simple message failed with exception: {e}")
    
    await asyncio.sleep(2)
    
    # Test 2: Health alert
    print("\n[Test 2] Sending mock health alert...")
    try:
        test_health = {
            "score": 72.5,
            "errors": ["Test error 1", "Test error 2"],
            "warnings": ["Test warning 1"]
        }
        result = await notif.send_health_alert(test_health)
        if result:
            print("✅ Health alert sent successfully")
        else:
            print("❌ Health alert failed (returned False)")
    except Exception as e:
        print(f"❌ Health alert failed with exception: {e}")
    
    await asyncio.sleep(2)
    
    # Test 3: Top picks notification (if available)
    print("\n[Test 3] Sending mock top picks...")
    try:
        test_picks = {
            "ETH": {
                "symbol": "ETH",
                "direction": "UP",
                "confidence": 85.0,
                "expected_move_pct": 4.2
            },
            "XRP": {
                "symbol": "XRP",
                "direction": "DOWN",
                "confidence": 78.0,
                "expected_move_pct": -3.5
            }
        }
        result = await notif.send_top10(test_picks)
        if result:
            print("✅ Top picks sent successfully")
        else:
            print("❌ Top picks failed (returned False)")
    except Exception as e:
        print(f"❌ Top picks failed with exception: {e}")
    
    print()
    print("=" * 60)
    print("✅ Telegram test complete!")
    print()
    print("Check your Telegram app for test messages.")
    print("If you didn't receive them:")
    print("  1. Verify TELEGRAM_BOT_TOKEN is correct")
    print("  2. Verify TELEGRAM_CHAT_ID is correct")
    print("  3. Make sure you've started a chat with the bot first")
    print("  4. Check bot has permission to send messages to the chat")
    
    return True


def main():
    """Run async tests."""
    asyncio.run(test_telegram_alerts())


if __name__ == "__main__":
    main()
