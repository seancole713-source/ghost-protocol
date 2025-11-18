#!/usr/bin/env python3
"""
Test Scheduled Predictions
Tests the prediction commands via Telegram
"""

import requests

BASE_URL = os.getenv("GHOST_BASE_URL", "http://localhost:8080")
CHAT_ID = "940596997"


def send_command(cmd):
    """Send command to Telegram webhook"""
    payload = {"message": {"chat": {"id": CHAT_ID}, "text": cmd}}

    print(f"\n{'=' * 60}")
    print(f"Testing: {cmd}")
    print(f"{'=' * 60}")

    try:
        response = requests.post(f"{BASE_URL}/telegram/webhook", json=payload, timeout=15)

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Command sent successfully")
            print("Check your Telegram for Ghost's response!")
        else:
            print(f"❌ Error: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")


print("\n" + "=" * 60)
print("🧪 TESTING SCHEDULED PREDICTIONS")
print("=" * 60)

# Test 1: Help command (should show prediction commands)
send_command("/help")

# Test 2: Force prediction
send_command("/predict")

# Test 3: Check prediction accuracy
send_command("/check")

print("\n" + "=" * 60)
print("✅ TEST COMPLETE")
print("=" * 60)
print("\nCheck your Telegram bot for Ghost's responses!")
print("You should see:")
print("  1. Updated help text with prediction commands")
print("  2. Pre-market prediction with current price & signal")
print("  3. Prediction accuracy check (comparison)")
