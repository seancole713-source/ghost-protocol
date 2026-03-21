#!/usr/bin/env python3
"""
Ghost Protocol - Health Alert System
Sends Telegram alerts when system health drops below threshold
"""

import os
import asyncio
import aiohttp
from datetime import datetime

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HEALTH_THRESHOLD = float(os.getenv("HEALTH_ALERT_THRESHOLD", "85.0"))  # Alert if health < 85
CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "300"))  # Check every 5 minutes

# API endpoint
HEALTH_API = "http://localhost:8000/integrity/audit/readonly"


async def send_telegram_alert(message: str):
    """Send alert to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram not configured (missing BOT_TOKEN or CHAT_ID)")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                if resp.status == 200:
                    print(f"✅ Telegram alert sent")
                    return True
                else:
                    print(f"❌ Telegram failed: {resp.status}")
                    return False
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False


async def check_system_health():
    """Check system health and send alerts if needed"""
    print(f"🔍 Checking system health...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(HEALTH_API, timeout=30) as resp:
                if resp.status != 200:
                    await send_telegram_alert(
                        "🚨 <b>Ghost Protocol Health Check FAILED</b>\n\n"
                        f"Health API returned status {resp.status}\n"
                        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
                    )
                    return
                
                data = await resp.json()
                
                if not data.get("ok"):
                    await send_telegram_alert(
                        "🚨 <b>Ghost Protocol Health Check FAILED</b>\n\n"
                        "Health API returned error\n"
                        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
                    )
                    return
                
                # Parse health score
                health_pct = data.get("score", 0)
                passed = data.get("passed", 0)
                failed = data.get("failed", 0)
                total = data.get("total", 0)
                
                print(f"📊 Health: {health_pct:.1f}/100 ({passed}/{total} checks passed)")
                
                # Alert if health below threshold
                if health_pct < HEALTH_THRESHOLD:
                    # Get failed checks
                    checks = data.get("checks", [])
                    failed_checks = [c for c in checks if not c.get("passed")]
                    
                    # Build alert message
                    alert_lines = [
                        f"🚨 <b>Ghost Protocol Health Alert</b>",
                        f"",
                        f"Health Score: <b>{health_pct:.1f}/100</b> (threshold: {HEALTH_THRESHOLD})",
                        f"Checks: {passed} passed, {failed} failed",
                        f"",
                        f"<b>Failed Checks:</b>"
                    ]
                    
                    for check in failed_checks[:10]:  # Show first 10 failures
                        name = check.get("name", "Unknown")
                        error = check.get("error", "No details")
                        alert_lines.append(f"❌ {name}: {error[:80]}")
                    
                    if len(failed_checks) > 10:
                        alert_lines.append(f"... and {len(failed_checks) - 10} more")
                    
                    alert_lines.append(f"")
                    alert_lines.append(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
                    
                    await send_telegram_alert("\n".join(alert_lines))
                else:
                    print(f"✅ Health OK: {health_pct:.1f}/100 (above {HEALTH_THRESHOLD} threshold)")
                
    except asyncio.TimeoutError:
        await send_telegram_alert(
            "🚨 <b>Ghost Protocol Health Check TIMEOUT</b>\n\n"
            "Health API did not respond within 30 seconds\n"
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )
    except Exception as e:
        await send_telegram_alert(
            f"🚨 <b>Ghost Protocol Health Check ERROR</b>\n\n"
            f"Error: {str(e)[:200]}\n"
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        print(f"❌ Health check error: {e}")


async def run_health_monitor():
    """Main monitoring loop"""
    print(f"🚀 Ghost Protocol Health Monitor started")
    print(f"📊 Threshold: {HEALTH_THRESHOLD}/100")
    print(f"⏰ Check interval: {CHECK_INTERVAL}s")
    print(f"")
    
    while True:
        try:
            await check_system_health()
        except Exception as e:
            print(f"❌ Monitor error: {e}")
        
        await asyncio.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        asyncio.run(run_health_monitor())
    except KeyboardInterrupt:
        print("\n🛑 Health monitor stopped")
