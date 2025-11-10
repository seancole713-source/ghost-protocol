#!/usr/bin/env python3
"""
Telegram Bot Repair Tool
Diagnoses and helps fix Telegram bot authentication issues
"""

import os
import sys

import requests
from dotenv import load_dotenv

print("=" * 80)
print("🔧 TELEGRAM BOT REPAIR TOOL")
print("=" * 80)
print()

# Load environment
load_dotenv("secrets.env")
current_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

print("📋 Current Configuration:")
print(f"   Token Length: {len(current_token)} chars")
print(
    f"   Token Preview: {current_token[:15]}...{current_token[-10:] if len(current_token) > 25 else ''}"
)
print(f"   Chat ID: {chat_id}")
print()

# Check token format
print("🔍 Token Format Analysis:")
if len(current_token) < 30:
    print("   ❌ Token too short (should be 40-50 chars)")
elif ":" not in current_token:
    print("   ❌ Invalid format (should contain ':')")
elif current_token.endswith("1234"):
    print("   ⚠️  Token looks like a placeholder (ends with 1234)")
else:
    print("   ✅ Token format looks valid")
print()

# Test current token
print("🧪 Testing Current Token:")
if current_token:
    try:
        response = requests.get(f"https://api.telegram.org/bot{current_token}/getMe", timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                bot_info = data.get("result", {})
                print("   ✅ TOKEN IS VALID!")
                print(f"   Bot Username: @{bot_info.get('username')}")
                print(f"   Bot Name: {bot_info.get('first_name')}")
                print(f"   Bot ID: {bot_info.get('id')}")
                print()
                print("✅ No fix needed - token is working!")
                sys.exit(0)
            else:
                print(f"   ❌ API returned error: {data}")
        elif response.status_code == 401:
            print("   ❌ UNAUTHORIZED - Token is invalid or revoked")
        elif response.status_code == 404:
            print("   ❌ NOT FOUND - Token format incorrect")
        else:
            print(f"   ❌ HTTP {response.status_code}: {response.text[:100]}")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
else:
    print("   ❌ No token found")

print()
print("=" * 80)
print("🔧 HOW TO FIX:")
print("=" * 80)
print()
print("1. Open Telegram and search for: @BotFather")
print()
print("2. Send this command to BotFather:")
print("   /mybots")
print()
print("3. Select your bot from the list (or create a new one with /newbot)")
print()
print("4. Click 'API Token' to view your token")
print()
print("5. Copy the token (format: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz)")
print()
print("6. Update secrets.env file:")
print("   TELEGRAM_BOT_TOKEN=<your_new_token_here>")
print()
print("7. Run this test again to verify:")
print("   python3 fix_telegram.py")
print()
print("=" * 80)
print()
print("📌 QUICK TEST:")
print("   If you have the token, test it with:")
print("   curl https://api.telegram.org/bot<TOKEN>/getMe")
print()
print("=" * 80)

# Offer to test a new token
print()
print("💡 TIP: You can also test a token right now by pasting it here.")
print("    (Press Enter to skip)")
print()
new_token = input("Enter new token to test (or press Enter): ").strip()

if new_token:
    print()
    print(f"Testing new token: {new_token[:15]}...{new_token[-10:]}")
    try:
        response = requests.get(f"https://api.telegram.org/bot{new_token}/getMe", timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                bot_info = data.get("result", {})
                print("✅ NEW TOKEN IS VALID!")
                print(f"   Bot Username: @{bot_info.get('username')}")
                print(f"   Bot Name: {bot_info.get('first_name')}")
                print()

                # Ask to save
                save = input("Save this token to secrets.env? (y/n): ").strip().lower()
                if save == "y":
                    # Read current secrets.env
                    with open("secrets.env") as f:
                        lines = f.readlines()

                    # Replace token line
                    with open("secrets.env", "w") as f:
                        for line in lines:
                            if line.startswith("TELEGRAM_BOT_TOKEN="):
                                f.write(f"TELEGRAM_BOT_TOKEN={new_token}\n")
                            else:
                                f.write(line)

                    print("✅ Token saved! Restart Ghost server to apply changes.")
            else:
                print(f"❌ Token invalid: {data}")
        else:
            print(f"❌ Token invalid (HTTP {response.status_code})")
    except Exception as e:
        print(f"❌ Error: {e}")

print()
print("=" * 80)
print("Need help? Check: https://core.telegram.org/bots#how-do-i-create-a-bot")
print("=" * 80)
