"""
📢 ALERT MANAGER
Clean Telegram formatting with hierarchical structure
Alert prioritization, cooldown logic, emoji system
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)

# Telegram config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Alert cooldowns (seconds)
_ALERT_COOLDOWNS: dict[str, float] = {}
_COOLDOWN_PERIOD = 300  # 5 minutes

# Priority levels
PRIORITY_CRITICAL = 1  # Immediate action required
PRIORITY_HIGH = 2      # Important update
PRIORITY_NORMAL = 3    # Informational
PRIORITY_LOW = 4       # Background info


# ============================================================================
# TELEGRAM FORMATTING
# ============================================================================

def format_daily_briefing(briefing: dict) -> str:
    """
    Format daily briefing with clean hierarchy
    """
    lines = []
    
    # Header
    lines.append("🌅 <b>GHOST PROTOCOL - DAILY BRIEFING</b>")
    lines.append(f"📅 {datetime.now().strftime('%A, %B %d, %Y')}")
    lines.append(f"⏰ 6:00 AM CT")
    lines.append("")
    
    # Market context
    regime = briefing.get("market_context", {}).get("regime", "UNKNOWN")
    regime_emoji = {
        "BULL": "🐂",
        "BEAR": "🐻",
        "CRASH": "💥",
        "RECOVERY": "📈",
        "SIDEWAYS": "↔️"
    }.get(regime, "❓")
    
    lines.append(f"{regime_emoji} <b>Market Regime:</b> {regime}")
    lines.append("")
    
    # Picks
    picks = briefing.get("picks", [])
    
    lines.append(f"🎯 <b>TODAY'S TOP {len(picks)} PICKS:</b>")
    lines.append("")
    
    for i, pick in enumerate(picks, 1):
        symbol = pick["symbol"]
        asset_type = pick["asset_type"]
        confidence = pick["confidence"]
        expected_gain = pick["expected_gain"]
        direction = pick["direction"]
        entry_low = pick["entry_low"]
        entry_high = pick["entry_high"]
        target = pick["target"]
        peak = pick["peak"]
        stop = pick["stop"]
        
        # Pick header
        asset_emoji = "📈" if asset_type == "stock" else "₿"
        direction_emoji = "🟢" if direction == "LONG" else "🔴"
        
        lines.append(f"{asset_emoji} <b>#{i}. ${symbol}</b> {direction_emoji}")
        
        # Confidence & Gain
        lines.append(f"├─ 💪 Confidence: <b>{confidence:.0f}%</b>")
        lines.append(f"├─ 🎯 Expected Gain: <b>+{expected_gain:.1f}%</b>")
        
        # Price levels
        lines.append(f"├─ 📍 Entry: ${entry_low:.2f} - ${entry_high:.2f}")
        lines.append(f"├─ 🎯 Target: ${target:.2f}")
        lines.append(f"├─ 🚀 Peak: ${peak:.2f}")
        lines.append(f"└─ 🛑 Stop: ${stop:.2f}")
        lines.append("")
    
    # Footer
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("🤖 <i>Ghost Protocol - Autonomous Predictions</i>")
    lines.append("📊 Real-time updates every 5 minutes")
    
    return "\n".join(lines)


def format_live_update(update: dict) -> str:
    """
    Format live position update with clean hierarchy
    """
    lines = []
    
    symbol = update["symbol"]
    action = update["action"]
    confidence = update["new_confidence"]
    expected_gain = update["new_expected_gain"]
    current_price = update["current_price"]
    pnl_pct = update["pnl_pct"]
    new_target = update["new_target"]
    new_peak = update["new_peak"]
    trail_stop = update["trail_stop"]
    reason = update.get("reason", "")
    
    # Action emoji
    action_emoji = {
        "EXIT": "🚨",
        "ADD": "✅",
        "TAKE_PROFITS": "💰",
        "STOP_HIT": "🛑",
        "HOLD": "⏳"
    }.get(action, "📊")
    
    # Header
    lines.append(f"{action_emoji} <b>{action}: ${symbol}</b>")
    lines.append("")
    
    # Current status
    pnl_emoji = "🟢" if pnl_pct > 0 else "🔴"
    lines.append(f"├─ 💵 Current Price: ${current_price:.2f}")
    lines.append(f"├─ {pnl_emoji} P&L: <b>{pnl_pct:+.1f}%</b>")
    lines.append("")
    
    # Updated metrics
    lines.append(f"├─ 💪 Confidence: {confidence:.0f}%")
    lines.append(f"├─ 🎯 Expected Gain: +{expected_gain:.1f}%")
    lines.append("")
    
    # Updated targets
    lines.append(f"├─ 🎯 New Target: ${new_target:.2f}")
    lines.append(f"├─ 🚀 New Peak: ${new_peak:.2f}")
    lines.append(f"└─ 🛑 Trail Stop: ${trail_stop:.2f}")
    
    if reason:
        lines.append("")
        lines.append(f"💡 <i>{reason}</i>")
    
    return "\n".join(lines)


def format_spike_alert(spike: dict) -> str:
    """
    Format spike detection alert
    """
    lines = []
    
    symbol = spike["symbol"]
    spike_type = spike["type"]
    change_pct = spike.get("change_pct", 0)
    volume_ratio = spike.get("volume_ratio", 0)
    
    # Spike emoji
    spike_emoji = {
        "PREMARKET_SPIKE": "🌅",
        "VOLUME_SPIKE": "📊",
        "NEWS_SPIKE": "📰",
        "SOCIAL_BUZZ": "🗣️"
    }.get(spike_type, "⚡")
    
    lines.append(f"{spike_emoji} <b>SPIKE DETECTED: ${symbol}</b>")
    
    if change_pct:
        change_emoji = "🟢" if change_pct > 0 else "🔴"
        lines.append(f"{change_emoji} Price: <b>{change_pct:+.1f}%</b>")
    
    if volume_ratio:
        lines.append(f"📊 Volume: <b>{volume_ratio:.1f}x average</b>")
    
    lines.append(f"🔍 Type: {spike_type.replace('_', ' ').title()}")
    
    return "\n".join(lines)


# ============================================================================
# ALERT SENDING
# ============================================================================

async def send_telegram_alert(message: str, priority: int = PRIORITY_NORMAL) -> bool:
    """
    Send alert to Telegram with priority handling
    """
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            LOGGER.warning("Telegram credentials not configured")
            return False
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            
            if resp.status_code != 200:
                LOGGER.error(f"Telegram alert failed: {resp.status_code} {resp.text}")
                return False
            
            LOGGER.info(f"📢 Telegram alert sent (priority {priority})")
            return True
            
    except Exception as e:
        LOGGER.error(f"Telegram alert failed: {e}")
        return False


async def send_daily_briefing(briefing: dict) -> bool:
    """
    Send daily briefing to Telegram
    """
    message = format_daily_briefing(briefing)
    return await send_telegram_alert(message, priority=PRIORITY_HIGH)


async def send_live_update(update: dict) -> bool:
    """
    Send live position update to Telegram
    """
    # Check cooldown
    symbol = update["symbol"]
    action = update["action"]
    cooldown_key = f"{symbol}_{action}"
    
    if cooldown_key in _ALERT_COOLDOWNS:
        if time.time() - _ALERT_COOLDOWNS[cooldown_key] < _COOLDOWN_PERIOD:
            LOGGER.debug(f"Alert cooldown active for {cooldown_key}")
            return False
    
    # Update cooldown
    _ALERT_COOLDOWNS[cooldown_key] = time.time()
    
    # Send alert
    message = format_live_update(update)
    
    # Priority based on action
    priority = {
        "EXIT": PRIORITY_CRITICAL,
        "STOP_HIT": PRIORITY_CRITICAL,
        "TAKE_PROFITS": PRIORITY_HIGH,
        "ADD": PRIORITY_HIGH,
        "HOLD": PRIORITY_LOW
    }.get(action, PRIORITY_NORMAL)
    
    return await send_telegram_alert(message, priority=priority)


async def send_spike_alert(spike: dict) -> bool:
    """
    Send spike detection alert to Telegram
    """
    # Check cooldown
    symbol = spike["symbol"]
    spike_type = spike["type"]
    cooldown_key = f"{symbol}_{spike_type}"
    
    if cooldown_key in _ALERT_COOLDOWNS:
        if time.time() - _ALERT_COOLDOWNS[cooldown_key] < _COOLDOWN_PERIOD:
            return False
    
    # Update cooldown
    _ALERT_COOLDOWNS[cooldown_key] = time.time()
    
    # Send alert
    message = format_spike_alert(spike)
    return await send_telegram_alert(message, priority=PRIORITY_HIGH)


# ============================================================================
# ALERT QUEUE
# ============================================================================

_ALERT_QUEUE: asyncio.Queue = asyncio.Queue()


async def queue_alert(alert: dict):
    """
    Add alert to queue for batch sending
    """
    await _ALERT_QUEUE.put(alert)


async def alert_processor_loop():
    """
    Background loop to process alert queue
    """
    LOGGER.info("🚀 Alert Processor: STARTED")
    
    while True:
        try:
            # Get alert from queue (wait up to 10 seconds)
            alert = await asyncio.wait_for(_ALERT_QUEUE.get(), timeout=10)
            
            # Send based on type
            alert_type = alert.get("type")
            
            if alert_type == "daily_briefing":
                await send_daily_briefing(alert["data"])
            
            elif alert_type == "live_update":
                await send_live_update(alert["data"])
            
            elif alert_type == "spike":
                await send_spike_alert(alert["data"])
            
            # Mark task done
            _ALERT_QUEUE.task_done()
            
            # Rate limit (max 30 messages/second per Telegram limits)
            await asyncio.sleep(0.1)
            
        except asyncio.TimeoutError:
            # No alerts in queue, continue
            continue
            
        except Exception as e:
            LOGGER.error(f"Alert processor error: {e}")
            await asyncio.sleep(1)
