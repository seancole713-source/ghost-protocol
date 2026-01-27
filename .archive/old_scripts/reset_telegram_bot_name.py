#!/usr/bin/env python3
"""
Reset Telegram Bot Name - Ghost Protocol
Resets bot display name and verifies configuration
"""

import os
import sys
import requests
from datetime import datetime

def get_bot_token():
    """Get bot token from environment"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN not found in environment")
        print("\nSet it with:")
        print("  export TELEGRAM_BOT_TOKEN='your_token_here'")
        sys.exit(1)
    return token

def get_bot_info(token):
    """Get current bot information"""
    try:
        url = f"https://api.telegram.org/bot{token}/getMe"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("ok"):
            print(f"❌ API Error: {data.get('description', 'Unknown error')}")
            return None
            
        return data.get("result", {})
    except Exception as e:
        print(f"❌ Failed to get bot info: {e}")
        return None

def set_bot_name(token, name="Ghost Protocol Bot"):
    """Reset bot display name"""
    try:
        url = f"https://api.telegram.org/bot{token}/setMyName"
        payload = {"name": name}
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("ok"):
            print(f"✅ Bot name successfully set to: '{name}'")
            return True
        else:
            print(f"❌ Failed to set name: {data.get('description', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Error setting bot name: {e}")
        return False

def set_bot_description(token, description="Ghost Protocol - AI-Powered Trading Signals"):
    """Set bot description"""
    try:
        url = f"https://api.telegram.org/bot{token}/setMyDescription"
        payload = {"description": description}
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("ok"):
            print(f"✅ Bot description set")
            return True
        else:
            print(f"⚠️  Failed to set description: {data.get('description', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"⚠️  Error setting description: {e}")
        return False

def main():
    print("=" * 60)
    print("🤖 TELEGRAM BOT NAME RESET TOOL")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Get token
    token = get_bot_token()
    print(f"🔑 Token: {token[:10]}...{token[-5:]}\n")
    
    # Get current bot info
    print("📋 Current Bot Information:")
    print("-" * 60)
    bot_info = get_bot_info(token)
    
    if not bot_info:
        print("❌ Could not retrieve bot information. Check your token.")
        sys.exit(1)
    
    print(f"   ID: {bot_info.get('id')}")
    print(f"   Username: @{bot_info.get('username')}")
    print(f"   Current Name: {bot_info.get('first_name')}")
    print(f"   Is Bot: {bot_info.get('is_bot')}")
    
    current_name = bot_info.get('first_name', '')
    
    # Check if name is suspicious
    suspicious_keywords = ['hack', 'hacked', 'pwned', 'owned', 'compromised']
    is_suspicious = any(keyword in current_name.lower() for keyword in suspicious_keywords)
    
    if is_suspicious:
        print(f"\n⚠️  ALERT: Suspicious bot name detected: '{current_name}'")
        print("   This may indicate unauthorized access.\n")
    
    # Reset name
    print("\n🔧 Resetting Bot Name:")
    print("-" * 60)
    
    desired_name = "Ghost Protocol Bot"
    if current_name == desired_name:
        print(f"✅ Bot name is already correct: '{desired_name}'")
    else:
        print(f"   Changing from: '{current_name}'")
        print(f"   Changing to:   '{desired_name}'")
        
        if set_bot_name(token, desired_name):
            print("   ✅ Name reset successful!")
        else:
            print("   ❌ Name reset failed!")
            sys.exit(1)
    
    # Set description
    print("\n📝 Setting Bot Description:")
    print("-" * 60)
    set_bot_description(token)
    
    # Verify changes
    print("\n✅ Verification:")
    print("-" * 60)
    updated_info = get_bot_info(token)
    if updated_info:
        print(f"   ID: {updated_info.get('id')}")
        print(f"   Username: @{updated_info.get('username')}")
        print(f"   Name: {updated_info.get('first_name')}")
        print(f"   ✅ Bot configuration verified!")
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE - Bot name has been reset")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Check Telegram to verify bot name appears correctly")
    print("  2. Send a test message via: python test_telegram_send.py")
    print("  3. Monitor bot name with: python monitor_telegram_bot.py")
    print()

if __name__ == "__main__":
    main()
