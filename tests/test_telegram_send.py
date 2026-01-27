#!/usr/bin/env python3
"""Test Telegram bot messaging"""

import os

import requests
from dotenv import load_dotenv

load_dotenv("secrets.env", override=True)
token = os.getenv("TELEGRAM_BOT_TOKEN", "")
chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

print("📤 Testing Telegram Message Sending")
print("=" * 60)
print(f"Chat ID: {chat_id}")
print()

# Test message
message = """🤖 Ghost Protocol Agent - System Test

✅ Telegram integration restored!

📊 All alert systems operational.
🎯 Ready for live trading intelligence.
⚡ Real-time alerts enabled."""

try:
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message},
        timeout=10,
    )

    if response.status_code == 200:
        data = response.json()
        if data.get("ok"):
            print("✅ TEST MESSAGE SENT SUCCESSFULLY!")
            print()
            print("Message Details:")
            result = data.get("result", {})
            print(f"  Message ID: {result.get('message_id')}")
            print(f"  Date: {result.get('date')}")
            print(f"  Chat: {result.get('chat', {}).get('id')}")
            print()
            print("🎉 Telegram alerting is fully operational!")
        else:
            print(f"❌ Send failed: {data}")
    else:
        print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")
