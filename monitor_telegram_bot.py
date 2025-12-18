#!/usr/bin/env python3
"""
Monitor Telegram Bot Name - Ghost Protocol
Detects unauthorized bot name changes and alerts
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from pathlib import Path

# Configuration
EXPECTED_BOT_NAME = "Ghost Protocol Bot"
CHECK_INTERVAL_SECONDS = 300  # 5 minutes
LOG_FILE = Path("logs/telegram_bot_monitor.log")
STATE_FILE = Path("data/telegram_bot_state.json")

def get_bot_token():
    """Get bot token from environment"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN not found in environment")
        sys.exit(1)
    return token

def get_bot_info(token):
    """Get current bot information"""
    try:
        url = f"https://api.telegram.org/bot{token}/getMe"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("ok"):
            return data.get("result", {})
        return None
    except Exception as e:
        log_message(f"❌ Error getting bot info: {e}")
        return None

def load_last_state():
    """Load last known bot state"""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        log_message(f"⚠️  Could not load state: {e}")
    return {}

def save_state(state):
    """Save current bot state"""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log_message(f"⚠️  Could not save state: {e}")

def log_message(message, level="INFO"):
    """Log message to file and console"""
    timestamp = datetime.now().isoformat()
    log_line = f"[{timestamp}] [{level}] {message}"
    
    print(log_line)
    
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a') as f:
            f.write(log_line + "\n")
    except Exception as e:
        print(f"⚠️  Could not write to log: {e}")

def send_alert(token, chat_id, message):
    """Send alert via Telegram"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json().get("ok", False)
    except Exception as e:
        log_message(f"⚠️  Could not send alert: {e}", "WARNING")
        return False

def check_bot_name(token, expected_name=EXPECTED_BOT_NAME):
    """Check if bot name matches expected value"""
    bot_info = get_bot_info(token)
    
    if not bot_info:
        log_message("❌ Could not retrieve bot info", "ERROR")
        return None
    
    current_name = bot_info.get('first_name', '')
    bot_id = bot_info.get('id')
    bot_username = bot_info.get('username')
    
    # Load last state
    last_state = load_last_state()
    last_name = last_state.get('name')
    last_check = last_state.get('last_check')
    
    # Current state
    current_state = {
        'name': current_name,
        'id': bot_id,
        'username': bot_username,
        'last_check': datetime.now().isoformat(),
        'check_count': last_state.get('check_count', 0) + 1
    }
    
    # Check for changes
    name_changed = last_name and last_name != current_name
    name_unauthorized = current_name != expected_name
    
    # Detect suspicious keywords
    suspicious_keywords = ['hack', 'hacked', 'pwned', 'owned', 'compromised', 'mishadox']
    is_suspicious = any(keyword in current_name.lower() for keyword in suspicious_keywords)
    
    status = {
        'timestamp': datetime.now().isoformat(),
        'bot_id': bot_id,
        'username': bot_username,
        'current_name': current_name,
        'expected_name': expected_name,
        'name_matches': current_name == expected_name,
        'name_changed': name_changed,
        'is_suspicious': is_suspicious,
        'last_name': last_name,
        'last_check': last_check
    }
    
    # Log status
    if name_changed:
        log_message(f"⚠️  BOT NAME CHANGED: '{last_name}' → '{current_name}'", "ALERT")
        current_state['last_change'] = datetime.now().isoformat()
        current_state['changes'] = last_state.get('changes', 0) + 1
    
    if is_suspicious:
        log_message(f"🚨 SUSPICIOUS BOT NAME DETECTED: '{current_name}'", "CRITICAL")
        
        # Send Telegram alert if chat ID available
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if chat_id:
            alert_msg = (
                f"🚨 <b>SECURITY ALERT</b>\n\n"
                f"Unauthorized bot name change detected!\n\n"
                f"Bot: @{bot_username}\n"
                f"Current Name: <code>{current_name}</code>\n"
                f"Expected: <code>{expected_name}</code>\n\n"
                f"Action Required: Run reset_telegram_bot_name.py"
            )
            send_alert(token, chat_id, alert_msg)
    
    elif name_unauthorized:
        log_message(f"⚠️  Bot name incorrect: '{current_name}' (expected: '{expected_name}')", "WARNING")
    else:
        log_message(f"✅ Bot name OK: '{current_name}'")
    
    # Save state
    save_state(current_state)
    
    return status

def monitor_continuous(interval=CHECK_INTERVAL_SECONDS):
    """Monitor bot name continuously"""
    token = get_bot_token()
    
    log_message("=" * 60)
    log_message("🤖 TELEGRAM BOT NAME MONITOR STARTED")
    log_message(f"Expected Name: {EXPECTED_BOT_NAME}")
    log_message(f"Check Interval: {interval}s")
    log_message("=" * 60)
    
    try:
        while True:
            status = check_bot_name(token)
            
            if status and not status['name_matches']:
                log_message(f"⚠️  Action needed: Bot name is '{status['current_name']}'", "WARNING")
                log_message(f"   Run: python reset_telegram_bot_name.py", "WARNING")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        log_message("\n👋 Monitor stopped by user")
        sys.exit(0)
    except Exception as e:
        log_message(f"❌ Monitor error: {e}", "ERROR")
        sys.exit(1)

def check_once():
    """Single check and report"""
    token = get_bot_token()
    status = check_bot_name(token)
    
    if not status:
        print("❌ Check failed")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("📊 BOT NAME STATUS REPORT")
    print("=" * 60)
    print(f"Bot ID:       {status['bot_id']}")
    print(f"Username:     @{status['username']}")
    print(f"Current Name: {status['current_name']}")
    print(f"Expected:     {status['expected_name']}")
    print(f"Status:       {'✅ OK' if status['name_matches'] else '⚠️  MISMATCH'}")
    
    if status['name_changed']:
        print(f"\n⚠️  Name changed from: {status['last_name']}")
    
    if status['is_suspicious']:
        print(f"\n🚨 SUSPICIOUS NAME DETECTED!")
        print(f"   Run: python reset_telegram_bot_name.py")
    
    print("=" * 60)
    
    return 0 if status['name_matches'] else 1

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        monitor_continuous()
    else:
        sys.exit(check_once())

if __name__ == "__main__":
    main()
