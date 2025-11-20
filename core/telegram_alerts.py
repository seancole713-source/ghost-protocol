"""
Ghost Telegram Alert Pipeline
Single source of truth for alert formatting with deduplication
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional, List, Dict
from zoneinfo import ZoneInfo

# Redis client (will be set by wolf_app.py)
REDIS_CLIENT = None
TELEGRAM_SEND_FUNC = None
TELEGRAM_CHAT_ID = None
LOGGER = None

# Timezone configuration
DEFAULT_TZ = "America/Chicago"

# Alert style configuration
ALERT_STYLE = os.getenv("ALERT_STYLE", "simple")  # Default to "simple" (Cash App style), was "verbose"
ALERT_SIMPLE_FORMAT = os.getenv("ALERT_SIMPLE_FORMAT", "cashapp")  # "cashapp" (default), "compact", "balanced", "context"
MIN_ALERT_CONFIDENCE = float(os.getenv("MIN_ALERT_CONFIDENCE", "0.60"))


@dataclass
class Alert:
    """Unified alert payload for all Ghost signals"""
    
    # Core identification
    symbol: str
    market: str  # "stock" or "crypto"
    
    # Signal data
    direction: Literal["BUY", "SELL", "HOLD", "WATCH"]
    confidence: float  # 0.0-1.0
    
    # Price information
    price_now: float
    price_prev: float
    change_pct: float
    
    # Prediction context
    predicted_pct: Optional[float] = None
    horizon_h: Optional[int] = None
    
    # Metadata
    source: str = "hunter"
    score: Optional[int] = None
    volume_ratio: Optional[float] = None
    provider: str = "polygon"
    
    # Factors
    factors: List[str] = field(default_factory=list)
    
    # Timestamps
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


def format_simple_alert(alert: Alert) -> str:
    """
    Format alert in CASH APP STYLE - Ultra clean, 3 lines max
    
    Args:
        alert: Unified alert payload
        
    Returns:
        Clean formatted string (NO markdown, plain text)
        
    Example:
        📈 SoundHound +12.63%
        Volume surge detected
        Confidence: 78%
        Action: BUY
    """
    # Format confidence as percentage
    conf_pct = int(alert.confidence * 100)
    
    # Arrow based on direction
    if alert.change_pct >= 0:
        arrow = "📈"
        sign = "+"
    else:
        arrow = "📉"
        sign = ""
    
    # First line: Symbol + % change (CASH APP STYLE)
    line1 = f"{arrow} {alert.symbol} {sign}{alert.change_pct:.2f}%"
    
    # Second line: Key insight (volume, momentum, breakout, etc.)
    if alert.volume_ratio and alert.volume_ratio >= 2.0:
        line2 = "Volume surge detected"
    elif abs(alert.change_pct) >= 10:
        line2 = "Strong momentum"
    elif alert.predicted_pct and abs(alert.predicted_pct) >= 5:
        line2 = f"Predicted move: {alert.predicted_pct:+.1f}%"
    elif alert.factors and len(alert.factors) > 0:
        line2 = alert.factors[0][:50]  # First factor, truncated
    else:
        line2 = "Price action detected"
    
    # Third line: Confidence
    line3 = f"Confidence: {conf_pct}%"
    
    # Fourth line: Action
    line4 = f"Action: {alert.direction}"
    
    return f"{line1}\n{line2}\n{line3}\n{line4}"


def format_prediction_alert_cashapp(
    symbol: str, 
    direction: str, 
    confidence: float, 
    price: float, 
    change_pct: float,
    horizon_h: int = 48
) -> str:
    """
    Format prediction alert in Cash App style
    
    Example:
        📈 WOLF +5.2%
        Ghost predicts: BUY
        Confidence: 78%
        Next 48h
    """
    arrow = "📈" if change_pct >= 0 else "📉"
    sign = "+" if change_pct >= 0 else ""
    conf_pct = int(confidence * 100)
    
    # Line 1: Symbol + Change
    line1 = f"{arrow} {symbol} {sign}{change_pct:.2f}%"
    
    # Line 2: Prediction
    line2 = f"Ghost predicts: {direction}"
    
    # Line 3: Confidence
    line3 = f"Confidence: {conf_pct}%"
    
    # Line 4: Horizon
    line4 = f"Next {horizon_h}h"
    
    return f"{line1}\n{line2}\n{line3}\n{line4}"


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
    
    Respects ALERT_STYLE env var: \"simple\" or \"verbose\" (default)

    Returns:
        True if alert was sent successfully
        False if skipped (duplicate) or failed
    """
    # Filter out 0% confidence (diagnostic only, not real predictions)
    confidence = prediction.get("confidence", 0)
    if confidence < 0.10:
        if LOGGER:
            LOGGER.info(f"Skipping 0% confidence alert: {market}/{symbol}/{horizon_bucket}")
        return False
    
    # Check minimum confidence threshold
    if confidence < MIN_ALERT_CONFIDENCE:
        if LOGGER:
            LOGGER.info(f"Skipping low confidence alert ({confidence:.0%} < {MIN_ALERT_CONFIDENCE:.0%}): {market}/{symbol}")
        return False
    
    # Check deduplication
    if not should_send_alert(market, symbol, horizon_bucket):
        if LOGGER:
            LOGGER.info(f"Skipping duplicate alert: {market}/{symbol}/{horizon_bucket}")
        return False

    # Render message based on ALERT_STYLE
    if ALERT_STYLE == "simple":
        # Build Alert DTO
        price = price_meta.get("price", 0.0)
        prev_close = price_meta.get("prev_close", 0.0)
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
        
        alert = Alert(
            symbol=symbol,
            market=market,
            direction=prediction.get("action", "HOLD"),
            confidence=confidence,
            price_now=price,
            price_prev=prev_close,
            change_pct=change_pct,
            horizon_h=prediction.get("horizon_h", 48),
            source="prediction",
            provider=price_meta.get("provider", "unknown"),
            factors=prediction.get("factors", [])
        )
        
        message = format_simple_alert(alert)
    else:
        # Use verbose format (existing)
        message = render_alert(symbol, market, horizon_bucket, prediction, price_meta, tz)

    # Send via Telegram
    if not TELEGRAM_SEND_FUNC or not TELEGRAM_CHAT_ID:
        if LOGGER:
            LOGGER.warning("Telegram not configured")
        return False

    try:
        TELEGRAM_SEND_FUNC(TELEGRAM_CHAT_ID, message)
        if LOGGER:
            LOGGER.info(f"Sent alert: {market}/{symbol}/{horizon_bucket} (style={ALERT_STYLE})")
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


def format_mover_alert_cashapp(symbol: str, price: float, change_pct: float, volume_mult: float = None) -> str:
    """
    Format mover alert in Cash App style
    
    Example:
        📈 BTC +8.5%
        Price: $43,251
        Volume: 3.2x avg
    """
    arrow = "📈" if change_pct >= 0 else "📉"
    sign = "+" if change_pct >= 0 else ""
    
    # Line 1: Symbol + Change
    line1 = f"{arrow} {symbol} {sign}{change_pct:.2f}%"
    
    # Line 2: Price
    if price >= 1000:
        price_str = f"${price:,.0f}"
    elif price >= 1:
        price_str = f"${price:.2f}"
    else:
        price_str = f"${price:.4f}"
    line2 = f"Price: {price_str}"
    
    # Line 3: Volume (if significant)
    if volume_mult and volume_mult >= 1.5:
        line3 = f"Volume: {volume_mult:.1f}x avg"
        return f"{line1}\n{line2}\n{line3}"
    else:
        return f"{line1}\n{line2}"


def send_mover_alert(kind: str, item: dict[str, Any]) -> bool:
    """
    Send a market mover alert to Telegram.
    
    Args:
        kind: "crypto" or "stocks"
        item: Mover dict with symbol, price, pct_1h, pct_24h, vol_mult, age_s, provider, tier
    
    Returns:
        True if sent successfully (or de-duped), False if error
    """
    if not TELEGRAM_SEND_FUNC or not TELEGRAM_CHAT_ID or not REDIS_CLIENT:
        if LOGGER:
            LOGGER.warning("Telegram or Redis not configured for mover alerts")
        return False
    
    try:
        symbol = item.get("symbol", "UNKNOWN")
        price = item.get("price", 0.0)
        pct_1h = item.get("pct_1h", 0.0)
        pct_24h = item.get("pct_24h", 0.0)
        vol_mult = item.get("vol_mult")
        age_s = item.get("age_s", 0)
        provider = item.get("provider", "unknown")
        tier = item.get("tier", "📊6+")
        
        # De-duplication key: ghost:alert:mover:{kind}:{symbol}:{tier}:{date}
        date = datetime.now(ZoneInfo(DEFAULT_TZ)).strftime("%Y-%m-%d")
        dedup_key = f"ghost:alert:mover:{kind}:{symbol}:{tier}:{date}"
        
        # Check if already sent today
        if REDIS_CLIENT.exists(dedup_key):
            if LOGGER:
                LOGGER.debug(f"Mover alert de-duped: {symbol} {tier} (already sent today)")
            return True  # Not an error, just already sent
        
        # Format volume multiplier
        vol_str = f"Vol×{vol_mult:.2f}" if vol_mult is not None else "Vol: N/A"
        
        # Use Cash App style format
        message = format_mover_alert_cashapp(
            symbol=symbol,
            price=price,
            change_pct=pct_24h,
            volume_mult=vol_mult
        )
        
        # Send via Telegram
        result = TELEGRAM_SEND_FUNC(message, TELEGRAM_CHAT_ID)
        
        if result:
            # Set de-dup key with 24h TTL
            REDIS_CLIENT.setex(dedup_key, 86400, "1")
            
            if LOGGER:
                LOGGER.info(f"Sent mover alert: {symbol} {tier}")
            
            return True
        else:
            if LOGGER:
                LOGGER.error(f"Failed to send mover alert: {symbol}")
            return False
            
    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Mover alert error: {e}", exc_info=True)
        return False
