#!/usr/bin/env python3
"""
Test Telegram Webhook Commands
Simulates Telegram bot commands to verify they work correctly
"""

import requests

BASE_URL = "http://localhost:8444"
CHAT_ID = "940596997"  # Your Telegram chat ID


def send_telegram_command(command: str):
    """Simulate a Telegram webhook update"""
    payload = {"message": {"chat": {"id": CHAT_ID}, "text": command}}

    print(f"\n{'=' * 60}")
    print(f"Testing: {command}")
    print(f"{'=' * 60}")

    try:
        response = requests.post(f"{BASE_URL}/telegram/webhook", json=payload, timeout=10)

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            print("✅ Command processed successfully")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def main():
    """Test all new crypto commands"""

    print("\n" + "=" * 60)
    print("🧪 TESTING TELEGRAM CRYPTO COMMANDS")
    print("=" * 60)

    # Test 1: Help command (should show new crypto section)
    print("\n📚 Test 1: Help Command")
    send_telegram_command("/help")

    # Test 2: List cryptos
    print("\n📋 Test 2: List Cryptos")
    send_telegram_command("/cryptos")

    # Test 3: Add new crypto
    print("\n➕ Test 3: Add Crypto (MATIC)")
    send_telegram_command("/watch MATIC")

    # Test 4: Try adding duplicate
    print("\n➕ Test 4: Add Duplicate (should warn)")
    send_telegram_command("/watch BTC")

    # Test 5: Remove crypto
    print("\n➖ Test 5: Remove Crypto (SHIB)")
    send_telegram_command("/unwatch SHIB")

    # Test 6: Try removing non-existent
    print("\n➖ Test 6: Remove Non-Existent (should warn)")
    send_telegram_command("/unwatch FAKE")

    # Test 7: List again to verify changes
    print("\n📋 Test 7: List Again (verify changes)")
    send_telegram_command("/cryptos")

    # Test 8: Natural language crypto question
    print("\n💬 Test 8: Natural Language Question")
    send_telegram_command("Should I buy PEPE? What's your 30-day prediction?")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 60)
    print("\nNOTE: Check Ghost Telegram bot for actual responses")
    print("The server should have sent messages to chat ID:", CHAT_ID)


if __name__ == "__main__":
    main()
