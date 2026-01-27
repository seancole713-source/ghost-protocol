#!/usr/bin/env python3
"""
Test Telegram directly
"""
import requests

TELEGRAM_BOT_TOKEN = "8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw"
TELEGRAM_CHAT_ID = "940596997"

# Test 1: Send simple message
url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
payload = {
    "chat_id": TELEGRAM_CHAT_ID,
    "text": "🎬 DEMO TEST: Ghost is testing Telegram connection!",
    "parse_mode": "Markdown"
}

print("Testing Telegram...")
response = requests.post(url, json=payload, timeout=10)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code == 200:
    print("✅ SUCCESS! Check your Telegram!")
else:
    print("❌ FAILED!")
