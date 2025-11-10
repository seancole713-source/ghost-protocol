"""
Ghost Telegram Alert Pipeline
Single source of truth for alert formatting with deduplication
"""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

# Redis client (will be set by wolf_app.py)
REDIS_CLIENT = None
TELEGRAM_SEND_FUNC = None
TELEGRAM_CHAT_ID = None
LOGGER = None

# Timezone configuration
DEFAULT_TZ = "America/Chicago"


def render_alert(
    symbol: str,
    market: str,
    horizon_bucket: str,
    prediction: dict[str, Any],
    price_meta: dict[str, Any],
    tz: str = DEFAULT_TZ,
) -> str:
    """
    Render a standardized alert message

    Args:
        symbol: Trading symbol (AAPL, BTC, etc.)
        market: "stock" or "crypto"
        horizon_bucket: "SHORT" (2h-30d) or "LONG" (30d-6m)
        prediction: {
            "action": "BUY" | "SELL" | "HOLD",
            "confidence": float (0.0-1.0),
            "direction": "UP" | "DOWN" | "HOLD",
            "factors": list[str]
        }
        price_meta: {
            "price": float,
            "prev_close": float,
            "provider": str,
            "after_hours": bool
        }
        tz: Timezone name (default: America/Chicago)

    Returns:
        Formatted alert message string
    """
    # Extract data
    action = prediction.get("action", "HOLD")
    confidence = prediction.get("confidence", 0.0)
    direction = prediction.get("direction", "HOLD")
    factors = prediction.get("factors", [])

    price = price_meta.get("price", 0.0)
    prev_close = price_meta.get("prev_close", 0.0)
    provider = price_meta.get("provider", "unknown")
    after_hours = price_meta.get("after_hours", False)

    # Handle low confidence - force HOLD
    if confidence < 0.10:
        action = "HOLD"
        direction = "HOLD"
        low_confidence_warning = "⚠️ <b>Low confidence</b> - model uncertain\n"
    else:
        low_confidence_warning = ""

    # Calculate price changes
    if prev_close > 0:
        delta = price - prev_close
        delta_pct = (delta / prev_close) * 100
    else:
        delta = 0.0
        delta_pct = 0.0

    # Format timestamps
    try:
        local_tz = ZoneInfo(tz)
        utc_tz = ZoneInfo("UTC")
        now_utc = datetime.now(utc_tz)
        now_local = now_utc.astimezone(local_tz)

        local_time = now_local.strftime("%Y-%m-%d %I:%M %p")
        utc_time = now_utc.strftime("%Y-%m-%d %H:%M")

        # Next check (2 hours from now for SHORT, 24 hours for LONG)
        if horizon_bucket == "SHORT":
            next_check_delta = 2 * 3600  # 2 hours
        else:
            next_check_delta = 24 * 3600  # 24 hours

        next_check_ts = now_local.timestamp() + next_check_delta
        next_check_local = datetime.fromtimestamp(next_check_ts, tz=local_tz).strftime("%I:%M %p")
    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Timezone error: {e}")
        local_time = datetime.now().strftime("%Y-%m-%d %I:%M %p")
        utc_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        next_check_local = "TBD"

    # Market emoji
    market_emoji = "📈" if market == "stock" else "₿"

    # Context label
    if market == "crypto":
        context = "CRYPTO"
    else:
        context = "STOCK"

    # Horizon label
    horizon_label = "Short-term (2h-30d)" if horizon_bucket == "SHORT" else "Long-term (30d-6m)"

    # After-hours tag
    after_hours_tag = " • After-hours (prev close)" if after_hours else ""

    # Confidence percentage (minimum 1% for display if non-zero)
    if confidence > 0:
        conf_pct = f"{max(1, round(confidence * 100))}%"
    else:
        conf_pct = "0%"

    # Build message
    message = f"""🌅 <b>{context} PREDICTION</b>
⏰ {local_time} CT | {utc_time} UTC

{market_emoji} {market.upper()} • {symbol}
Price: ${price:.2f}  Prev: ${prev_close:.2f}  Δ: {delta:+.2f} ({delta_pct:+.2f}%)
Provider: {provider}{after_hours_tag}

{low_confidence_warning}🎯 <b>{horizon_label}</b>
Action: <b>{action}</b>   Confidence: {conf_pct}   Direction: {direction}

📈 <b>Factors:</b>
"""

    # Add factors (max 5)
    if factors:
        for factor in factors[:5]:
            message += f"• {factor}\n"
    else:
        message += "• No factors available\n"

    message += f"""
🔁 Next check: {next_check_local} CT
"""

    return message


def should_send_alert(market: str, symbol: str, horizon: str) -> bool:
    """
    Check if alert should be sent (deduplication check)

    Uses Redis SET with 24h TTL to prevent duplicate alerts
    Key format: alerts:sent:{market}:{symbol}:{horizon}:{date}

    Returns:
        True if alert should be sent (not sent yet today)
        False if alert already sent today
    """
    if not REDIS_CLIENT:
        # No Redis - always send (fallback)
        return True

    try:
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")

        # Build dedup key
        dedup_key = f"alerts:sent:{market}:{symbol}:{horizon}:{today}"

        # Try to add to set (returns 1 if added, 0 if exists)
        result = REDIS_CLIENT.sadd(dedup_key, "1")

        # Set 24h expiration if this is first time
        if result == 1:
            REDIS_CLIENT.expire(dedup_key, 24 * 3600)
            return True
        else:
            return False

    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Dedup check error: {e}")
        # On error, allow send (fail open)
        return True


def send_alert(
    symbol: str,
    market: str,
    horizon_bucket: str,
    prediction: dict[str, Any],
    price_meta: dict[str, Any],
    tz: str = DEFAULT_TZ,
) -> bool:
    """
    Send alert via Telegram with deduplication

    Returns:
        True if alert was sent successfully
        False if skipped (duplicate) or failed
    """
    # Check deduplication
    if not should_send_alert(market, symbol, horizon_bucket):
        if LOGGER:
            LOGGER.info(f"Skipping duplicate alert: {market}/{symbol}/{horizon_bucket}")
        return False

    # Render message
    message = render_alert(symbol, market, horizon_bucket, prediction, price_meta, tz)

    # Send via Telegram
    if not TELEGRAM_SEND_FUNC or not TELEGRAM_CHAT_ID:
        if LOGGER:
            LOGGER.warning("Telegram not configured")
        return False

    try:
        TELEGRAM_SEND_FUNC(TELEGRAM_CHAT_ID, message)
        if LOGGER:
            LOGGER.info(f"Sent alert: {market}/{symbol}/{horizon_bucket}")
        return True
    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Failed to send alert: {e}")
        return False


def get_recent_alerts(limit: int = 20) -> list[dict[str, Any]]:
    """
    Get recent alert envelopes for verification

    Returns:
        List of alert metadata (without full message content)
    """
    if not REDIS_CLIENT:
        return []

    try:
        # Scan for alert keys
        alerts = []
        cursor = 0
        pattern = "alerts:sent:*"

        while True:
            cursor, keys = REDIS_CLIENT.scan(cursor, match=pattern, count=100)
            for key in keys:
                # Parse key format: alerts:sent:{market}:{symbol}:{horizon}:{date}
                parts = key.decode() if isinstance(key, bytes) else key
                parts = parts.split(":")
                if len(parts) >= 6:
                    alerts.append(
                        {
                            "market": parts[2],
                            "symbol": parts[3],
                            "horizon": parts[4],
                            "date": parts[5],
                        }
                    )

            if cursor == 0:
                break

        # Sort by date (newest first) and limit
        alerts.sort(key=lambda x: x["date"], reverse=True)
        return alerts[:limit]

    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Failed to get recent alerts: {e}")
        return []
