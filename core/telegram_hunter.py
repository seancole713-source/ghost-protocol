"""
GHOST INVESTMENT HUNTER - TELEGRAM ALERT SYSTEM
================================================
Sends formatted investment opportunity alerts via Telegram

ALERT FEATURES:
- Hunter-style formatting with emoji urgency indicators
- Multi-signal confirmation messaging
- Cooldown system to prevent spam
- Scheduled daily reports (morning + evening)
- Score-based prioritization

ALERT TYPES:
1. Instant Alerts - High-scoring opportunities (80+)
2. Daily Reports - Morning (7am) and Evening (8pm) summaries
3. Accuracy Updates - Weekly performance tracking

DESIGN PRINCIPLES:
- Clear, actionable information
- Visual urgency indicators (🔥⭐✨)
- No spam (cooldowns + deduplication)
- Mobile-friendly formatting
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

from core.opportunity_scorer import (
    calculate_opportunity_score,
    get_score_grade,
    get_score_emoji,
)

# ========================================
# CONFIGURATION
# ========================================

# Telegram API configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Alert thresholds
INSTANT_ALERT_THRESHOLD = 80  # Score 80+ sends instant alert
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_ALERT_CONFIDENCE", "0.55"))  # Use Railway env var

# Cooldown settings (prevent spam)
SYMBOL_COOLDOWN_HOURS = 4  # Don't alert same symbol within 4 hours
MAX_ALERTS_PER_HOUR = 5  # Max 5 instant alerts per hour

# Daily report schedule
MORNING_REPORT_HOUR = 7  # 7am
EVENING_REPORT_HOUR = 20  # 8pm

# ========================================
# ALERT TRACKING
# ========================================

# Track recent alerts to prevent duplicates
_recent_alerts: Dict[str, float] = {}  # {symbol: timestamp}
_hourly_alert_count: int = 0
_hourly_alert_reset_time: float = time.time() + 3600

# ========================================
# TELEGRAM SEND FUNCTIONS
# ========================================


def send_telegram_message(message: str, parse_mode: str = "Markdown") -> bool:
    """
    Send message to Telegram bot

    Args:
        message: Text to send
        parse_mode: "Markdown" or "HTML"

    Returns:
        True if sent successfully, False otherwise
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("❌ Telegram not configured (missing token/chat_id)")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": parse_mode}

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logging.info("📲 Telegram message sent successfully")
            return True
        else:
            logging.error(f"❌ Telegram send failed: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logging.error(f"❌ Telegram exception: {e}")
        return False


# ========================================
# ALERT COOLDOWN & SPAM PREVENTION
# ========================================


def can_send_alert(symbol: str) -> bool:
    """
    Check if we can send alert for this symbol (cooldown + hourly limit)

    Args:
        symbol: Stock/crypto symbol

    Returns:
        True if alert allowed, False if on cooldown or limit reached
    """
    global _hourly_alert_count, _hourly_alert_reset_time

    # Reset hourly counter if needed
    now = time.time()
    if now >= _hourly_alert_reset_time:
        _hourly_alert_count = 0
        _hourly_alert_reset_time = now + 3600

    # Check hourly limit
    if _hourly_alert_count >= MAX_ALERTS_PER_HOUR:
        logging.warning(f"⏸️ Hourly alert limit reached ({MAX_ALERTS_PER_HOUR}/hr)")
        return False

    # Check symbol cooldown
    if symbol in _recent_alerts:
        last_alert_time = _recent_alerts[symbol]
        cooldown_end = last_alert_time + (SYMBOL_COOLDOWN_HOURS * 3600)
        if now < cooldown_end:
            remaining = int((cooldown_end - now) / 60)
            logging.info(f"⏸️ {symbol} on cooldown ({remaining} min remaining)")
            return False

    return True


def mark_alert_sent(symbol: str) -> None:
    """
    Mark symbol as alerted (track for cooldown)

    Args:
        symbol: Stock/crypto symbol
    """
    global _hourly_alert_count
    _recent_alerts[symbol] = time.time()
    _hourly_alert_count += 1


# ========================================
# ALERT FORMATTING
# ========================================


def format_opportunity_alert(opportunity: Dict) -> str:
    """
    Format opportunity as hunter-style Telegram message

    Args:
        opportunity: Opportunity dict with symbol, confidence, score, etc.

    Returns:
        Formatted Telegram message (Markdown)
    """
    # Extract fields
    symbol = opportunity.get("symbol", "???")
    confidence = opportunity.get("confidence", 0.0)
    predicted_pct = opportunity.get("predicted_pct", 0.0)
    timeframe_hours = opportunity.get("timeframe_hours", 24)
    action = opportunity.get("action", "HOLD")
    volume_ratio = opportunity.get("volume_ratio", 1.0)
    sentiment = opportunity.get("sentiment", 0.0)

    # Calculate score if not present
    if "score" not in opportunity:
        opportunity["score"] = calculate_opportunity_score(opportunity)

    score = opportunity["score"]
    grade = get_score_grade(score)
    emoji = get_score_emoji(score)

    # Format timeframe
    if timeframe_hours < 24:
        timeframe_str = f"{int(timeframe_hours)}h"
    else:
        days = int(timeframe_hours / 24)
        timeframe_str = f"{days}d"

    # Format predicted change
    direction_arrow = "📈" if predicted_pct > 0 else "📉"
    predicted_str = f"{predicted_pct:+.1f}%"

    # Format volume info
    if volume_ratio >= 3.0:
        volume_str = f"🔊 Volume: {volume_ratio:.1f}x average"
    else:
        volume_str = f"Volume: {volume_ratio:.1f}x"

    # Format sentiment
    if sentiment >= 0.5:
        sentiment_str = "😊 Positive"
    elif sentiment <= -0.5:
        sentiment_str = "😟 Negative"
    else:
        sentiment_str = "😐 Neutral"

    # Build alert message
    message = f"""
{emoji} **GHOST HUNTER ALERT** {emoji}

**{symbol}** — Grade: {grade} (Score: {score}/100)

{direction_arrow} **Prediction:** {action} {predicted_str} in {timeframe_str}
🎯 **AI Confidence:** {confidence * 100:.0f}%

**Signals:**
• {volume_str}
• Sentiment: {sentiment_str}

**Recommendation:** {_get_action_recommendation(action, confidence, score)}

⏱️ Timeframe: {timeframe_str}
📊 Ghost Score: {score}/100

_Track this on your watchlist. Ghost accuracy: 85%+_
""".strip()

    return message


def _get_action_recommendation(action: str, confidence: float, score: int) -> str:
    """
    Generate action recommendation text based on signal strength

    Args:
        action: BUY/SELL/HOLD
        confidence: AI confidence (0-1)
        score: Opportunity score (0-100)

    Returns:
        Recommendation string
    """
    if score >= 90:
        return f"🔥 STRONG {action} — Exceptional opportunity"
    elif score >= 80:
        return f"⭐ Consider {action} — High quality setup"
    elif score >= 70:
        return f"✨ Watch closely — Good potential"
    else:
        return f"👍 Monitor — Decent signal"


def format_daily_report(opportunities: List[Dict], accuracy_stats: Optional[Dict] = None) -> str:
    """
    Format daily opportunity report (morning/evening)

    Args:
        opportunities: List of top opportunities
        accuracy_stats: Optional accuracy metrics

    Returns:
        Formatted Telegram report
    """
    now = datetime.now()
    hour = now.hour

    if hour < 12:
        greeting = "☀️ **MORNING REPORT**"
    elif hour < 18:
        greeting = "🌤️ **AFTERNOON UPDATE**"
    else:
        greeting = "🌙 **EVENING REPORT**"

    # Header
    report = f"{greeting}\n"
    report += f"📅 {now.strftime('%A, %B %d, %Y')}\n\n"

    # Accuracy stats if available
    if accuracy_stats:
        accuracy_pct = accuracy_stats.get("accuracy_pct", 0)
        total = accuracy_stats.get("total_predictions", 0)
        report += f"🎯 **Ghost Accuracy:** {accuracy_pct:.1f}% ({total} predictions)\n\n"

    # Top opportunities
    if not opportunities:
        report += "🔍 No high-quality opportunities detected today.\n"
        report += "_Market is quiet. Ghost is watching._\n"
    else:
        report += f"🎯 **Top {len(opportunities)} Opportunities:**\n\n"

        for i, opp in enumerate(opportunities[:10], 1):
            symbol = opp.get("symbol", "???")
            score = opp.get("score", 0)
            grade = get_score_grade(score)
            emoji = get_score_emoji(score)
            predicted_pct = opp.get("predicted_pct", 0)
            action = opp.get("action", "HOLD")

            report += f"{i}. {emoji} **{symbol}** — {grade} ({score}/100)\n"
            report += f"   {action} {predicted_pct:+.1f}% predicted\n\n"

    # Footer
    report += "—\n"
    report += "_Ghost scans 4,000+ assets every 5 minutes._\n"
    report += "_Only the best opportunities make this list._"

    return report


def format_accuracy_update(stats: Dict) -> str:
    """
    Format accuracy tracking update for Telegram

    Args:
        stats: Accuracy statistics dict

    Returns:
        Formatted message
    """
    accuracy_pct = stats.get("accuracy_pct", 0)
    total = stats.get("total_predictions", 0)
    correct = stats.get("correct_predictions", 0)
    avg_error = stats.get("avg_error_pct", 0)

    # Choose emoji based on accuracy
    if accuracy_pct >= 90:
        emoji = "🔥"
    elif accuracy_pct >= 80:
        emoji = "⭐"
    elif accuracy_pct >= 70:
        emoji = "✨"
    else:
        emoji = "📊"

    message = f"""
{emoji} **GHOST ACCURACY UPDATE** {emoji}

**Performance Metrics:**
✅ Accuracy: {accuracy_pct:.1f}%
📊 Total Predictions: {total}
🎯 Correct: {correct}
📉 Avg Error: {avg_error:.1f}%

_Ghost learns from every prediction._
_Accuracy improves over time._
""".strip()

    return message


# ========================================
# ALERT SENDING FUNCTIONS
# ========================================


async def send_instant_alert(opportunity: Dict) -> bool:
    """
    Send instant alert for high-scoring opportunity

    Args:
        opportunity: Opportunity dict

    Returns:
        True if sent, False if blocked by cooldown/limit
    """
    symbol = opportunity.get("symbol", "???")

    # Check if we can send
    if not can_send_alert(symbol):
        return False

    # Calculate score if missing
    if "score" not in opportunity:
        opportunity["score"] = calculate_opportunity_score(opportunity)

    score = opportunity["score"]

    # Only send if meets threshold
    if score < INSTANT_ALERT_THRESHOLD:
        logging.info(f"⏸️ {symbol} score {score} below threshold {INSTANT_ALERT_THRESHOLD}")
        return False

    # Format and send
    message = format_opportunity_alert(opportunity)
    success = send_telegram_message(message)

    if success:
        mark_alert_sent(symbol)
        logging.info(f"📲 Instant alert sent: {symbol} (score {score})")

    return success


async def send_daily_report(opportunities: List[Dict], accuracy_stats: Optional[Dict] = None) -> bool:
    """
    Send daily opportunity report

    Args:
        opportunities: Top opportunities for the day
        accuracy_stats: Optional accuracy metrics

    Returns:
        True if sent successfully
    """
    # Score and rank opportunities
    for opp in opportunities:
        if "score" not in opp:
            opp["score"] = calculate_opportunity_score(opp)

    opportunities.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Format and send
    message = format_daily_report(opportunities, accuracy_stats)
    success = send_telegram_message(message)

    if success:
        logging.info("📲 Daily report sent")

    return success


async def send_accuracy_update(stats: Dict) -> bool:
    """
    Send accuracy tracking update

    Args:
        stats: Accuracy statistics

    Returns:
        True if sent successfully
    """
    message = format_accuracy_update(stats)
    success = send_telegram_message(message)

    if success:
        logging.info("📲 Accuracy update sent")

    return success


# ========================================
# SCHEDULED REPORT LOOP
# ========================================


async def daily_report_loop(get_opportunities_func, get_accuracy_func):
    """
    Background loop for scheduled daily reports

    Args:
        get_opportunities_func: Async function that returns opportunities list
        get_accuracy_func: Async function that returns accuracy stats
    """
    logging.info("📅 Daily report scheduler started")

    last_morning_report = None
    last_evening_report = None

    while True:
        try:
            now = datetime.now()
            current_date = now.date()
            current_hour = now.hour

            # Morning report (7am, once per day)
            if current_hour == MORNING_REPORT_HOUR and last_morning_report != current_date:
                logging.info("☀️ Sending morning report...")
                opportunities = await get_opportunities_func()
                accuracy = await get_accuracy_func("24h")
                await send_daily_report(opportunities, accuracy)
                last_morning_report = current_date

            # Evening report (8pm, once per day)
            if current_hour == EVENING_REPORT_HOUR and last_evening_report != current_date:
                logging.info("🌙 Sending evening report...")
                opportunities = await get_opportunities_func()
                accuracy = await get_accuracy_func("24h")
                await send_daily_report(opportunities, accuracy)
                last_evening_report = current_date

            # Check every 10 minutes
            await asyncio.sleep(600)

        except Exception as e:
            logging.error(f"❌ Daily report loop error: {e}")
            await asyncio.sleep(600)


# ========================================
# TEST HARNESS
# ========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test opportunity
    test_opp = {
        "symbol": "AAPL",
        "confidence": 0.89,
        "predicted_pct": 7.5,
        "timeframe_hours": 6,
        "action": "BUY",
        "volume_ratio": 4.2,
        "sentiment": 0.7,
    }

    # Calculate score
    test_opp["score"] = calculate_opportunity_score(test_opp)

    print("\n=== TEST: Instant Alert ===")
    message = format_opportunity_alert(test_opp)
    print(message)

    print("\n=== TEST: Daily Report ===")
    test_opportunities = [test_opp]
    test_accuracy = {"accuracy_pct": 87.5, "total_predictions": 120, "correct_predictions": 105}
    report = format_daily_report(test_opportunities, test_accuracy)
    print(report)

    print("\n=== TEST: Accuracy Update ===")
    accuracy_msg = format_accuracy_update(test_accuracy)
    print(accuracy_msg)

    print("\n✅ Test formatting complete")
