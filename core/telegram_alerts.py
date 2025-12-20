"""
Ghost Telegram Alert Pipeline
Single source of truth for alert formatting with deduplication

SMART CAP SYSTEM (v2.0):
- Max 10 predictions per day (configurable via DAILY_ALERT_CAP)
- Only the BEST predictions get sent (highest confidence)
- Raises minimum confidence to 80% (configurable via MIN_ALERT_CONFIDENCE)
- Tracks daily counts to prevent spam
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Literal, Optional, List, Dict
from zoneinfo import ZoneInfo
import threading

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
MIN_ALERT_CONFIDENCE = float(os.getenv("MIN_ALERT_CONFIDENCE", "0.80"))  # RAISED: 80% minimum (was 70%)

# ============================================================================
# SMART CAP SYSTEM - Prevents spam, only sends best predictions
# ============================================================================
DAILY_ALERT_CAP = int(os.getenv("DAILY_ALERT_CAP", "10"))  # Max 10 alerts per day
SMART_CAP_ENABLED = os.getenv("SMART_CAP_ENABLED", "1") == "1"  # Enable by default

# Thread-safe daily tracking
_daily_lock = threading.Lock()
_daily_alerts_sent: Dict[str, int] = {}  # {date_str: count}
_daily_alerts_log: Dict[str, List[Dict]] = {}  # {date_str: [{symbol, confidence, timestamp}]}


def _get_today_key() -> str:
    """Get today's date key in Chicago timezone"""
    try:
        return datetime.now(ZoneInfo(DEFAULT_TZ)).strftime("%Y-%m-%d")
    except:
        return datetime.now().strftime("%Y-%m-%d")


def get_daily_alert_count() -> int:
    """Get how many alerts have been sent today"""
    with _daily_lock:
        today = _get_today_key()
        return _daily_alerts_sent.get(today, 0)


def get_daily_alert_log() -> List[Dict]:
    """Get list of alerts sent today"""
    with _daily_lock:
        today = _get_today_key()
        return _daily_alerts_log.get(today, []).copy()


def _increment_daily_count(symbol: str, confidence: float) -> bool:
    """
    Increment daily alert count. Returns True if under cap, False if capped.
    """
    with _daily_lock:
        today = _get_today_key()
        
        # Clean up old days (keep only today)
        old_keys = [k for k in _daily_alerts_sent.keys() if k != today]
        for k in old_keys:
            _daily_alerts_sent.pop(k, None)
            _daily_alerts_log.pop(k, None)
        
        current_count = _daily_alerts_sent.get(today, 0)
        
        if current_count >= DAILY_ALERT_CAP:
            return False
        
        # Increment and log
        _daily_alerts_sent[today] = current_count + 1
        
        if today not in _daily_alerts_log:
            _daily_alerts_log[today] = []
        _daily_alerts_log[today].append({
            "symbol": symbol,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "count": current_count + 1
        })
        
        return True


def check_smart_cap(symbol: str, confidence: float) -> tuple[bool, str]:
    """
    Check if this alert should be sent based on smart cap rules.
    
    Returns:
        (allowed, reason) - True if allowed to send, with reason string
    """
    if not SMART_CAP_ENABLED:
        return True, "smart_cap_disabled"
    
    today = _get_today_key()
    current_count = get_daily_alert_count()
    
    # Check if at cap
    if current_count >= DAILY_ALERT_CAP:
        return False, f"daily_cap_reached ({current_count}/{DAILY_ALERT_CAP})"
    
    # Check minimum confidence (raised to 80%)
    if confidence < MIN_ALERT_CONFIDENCE:
        return False, f"below_min_confidence ({confidence:.0%} < {MIN_ALERT_CONFIDENCE:.0%})"
    
    # Additional quality gates for later slots
    remaining = DAILY_ALERT_CAP - current_count
    
    # Last 3 slots require 85%+ confidence
    if remaining <= 3 and confidence < 0.85:
        return False, f"reserved_for_high_conviction (need 85%+, got {confidence:.0%})"
    
    # Last slot requires 90%+ confidence
    if remaining == 1 and confidence < 0.90:
        return False, f"last_slot_reserved (need 90%+, got {confidence:.0%})"
    
    return True, f"allowed ({current_count + 1}/{DAILY_ALERT_CAP})"


def format_daily_digest_cashapp(
    *,
    as_of_iso: str,
    window_label: str,
    picks: list[dict[str, Any]],
    tz: str = DEFAULT_TZ,
) -> str:
    """Cash-App-style daily list.

    Each pick dict should include:
      symbol, predicted_pct (float), confidence (0..1), why (str)
    """
    try:
        local_tz = ZoneInfo(tz)
    except Exception:
        local_tz = None

    header = f"GHOST — Daily Picks\nAs of: {as_of_iso}\nWindow: {window_label}"
    lines: list[str] = [header, ""]

    if not picks:
        lines.append("No picks (gate closed): live data/news unavailable OR <70% confidence.")
        return "\n".join(lines)

    for p in picks:
        sym = str(p.get("symbol") or "").upper().strip()
        pred = p.get("predicted_pct")
        conf = p.get("confidence")
        why = str(p.get("why") or "").strip()

        try:
            pred_f = float(pred)
            pred_s = f"{pred_f:+.2f}%"
        except Exception:
            pred_s = "?%"

        try:
            conf_f = float(conf)
            conf_s = f"{int(round(conf_f * 100)):d}%"
        except Exception:
            conf_s = "?%"

        # One-line per pick, then one "why" line.
        lines.append(f"{sym}  {pred_s}  ({conf_s})")
        if why:
            lines.append(f"- {why[:160]}")

    return "\n".join(lines)


def _fmt_price(v: Any) -> str:
    try:
        fv = float(v)
        if fv == 0.0:
            return "$0.00"
        if abs(fv) >= 100:
            return f"${fv:,.2f}"
        return f"${fv:,.4f}"
    except Exception:
        return "?"


def _pct(a: Any, b: Any) -> float | None:
    try:
        fa = float(a)
        fb = float(b)
        if fb == 0:
            return None
        return (fa - fb) / fb * 100.0
    except Exception:
        return None


def format_touch_target_signal(
    *,
    symbol: str,
    prediction: dict[str, Any],
    price_meta: dict[str, Any],
) -> str:
    """Format a prediction into the touch-target + gating spec (plain text).

    Uses:
      - target is "hit" if price touches within horizon window AND direction is correct
      - Stage5 = calibrated touch >= 0.70 at ±1.0%
      - Stage6 = calibrated touch >= 0.70 at ±0.5%
    """
    # Core fields
    horizon_h = int(prediction.get("horizon_h") or 48)
    confidence = float(prediction.get("confidence") or 0.0)

    expected_move_pct = prediction.get("expected_move_pct")
    try:
        expected_move_pct_f = None if expected_move_pct is None else float(expected_move_pct)
    except Exception:
        expected_move_pct_f = None

    action = str(prediction.get("action") or "HOLD").upper()
    direction = str(prediction.get("direction") or "").upper()
    if not direction:
        direction = "UP" if action == "BUY" else "DOWN" if action == "SELL" else "FLAT"

    dir_emoji = "🟢" if direction == "UP" else "🔴" if direction == "DOWN" else "⚪"

    # Prices
    entry_price = prediction.get("entry_price")
    if entry_price is None:
        entry_price = price_meta.get("price")

    target_price = (
        prediction.get("target_price")
        if prediction.get("target_price") is not None
        else prediction.get("take_profit")
        if prediction.get("take_profit") is not None
        else prediction.get("take_profit_price")
    )
    stop_loss = (
        prediction.get("stop_loss")
        if prediction.get("stop_loss") is not None
        else prediction.get("stop_loss_price")
    )

    target_pct = _pct(target_price, entry_price)
    stop_pct = _pct(stop_loss, entry_price)

    # Calibration / gating
    stage5_ok = bool(prediction.get("stage5_ok"))
    stage6_ok = bool(prediction.get("stage6_ok"))
    gate = str(prediction.get("gate") or "").upper()
    if not gate:
        gate = "EXECUTION" if stage6_ok else "ANALYSIS" if stage5_ok else "MONITOR"

    p1 = prediction.get("touch_calibrated_1pct")
    p05 = prediction.get("touch_calibrated_0_5pct")
    try:
        p1s = "?" if p1 is None else f"{float(p1):.0%}"
    except Exception:
        p1s = "?"
    try:
        p05s = "?" if p05 is None else f"{float(p05):.0%}"
    except Exception:
        p05s = "?"

    s5 = "✅ Stage5 OK" if stage5_ok else "❌ Stage5 Not OK"
    s6 = "✅ Stage6 OK" if stage6_ok else "❌ Stage6 Not OK"

    # Human-facing risk label based on calibrated win prob when present.
    # Use execution tier if known; otherwise fall back to analysis tier; otherwise raw confidence.
    risk_p = None
    for cand in (p05, p1, confidence):
        try:
            if cand is None:
                continue
            risk_p = float(cand)
            break
        except Exception:
            continue

    risk_label = "UNKNOWN"
    if risk_p is not None:
        if risk_p >= 0.85:
            risk_label = "LOW RISK (85%+)"
        elif risk_p >= 0.70:
            risk_label = "OK (>=70%)"
        elif risk_p >= 0.45:
            risk_label = "HIGH RISK (<70%)"
        else:
            risk_label = "VERY HIGH RISK (<45%)"

    require_stage6 = os.getenv("AUTO_EXECUTION_REQUIRE_STAGE6", "1").strip() not in ("0", "false", "False")
    auto_status = "ENABLED (Stage6)" if (not require_stage6 or stage6_ok) else "BLOCKED (Stage6 required)"

    pid = prediction.get("prediction_id") or prediction.get("id") or ""
    pid_line = f"ID: {pid}" if pid else None

    lines = [
        f"🚦 GHOST SIGNAL — {gate}",
        f"{symbol} {dir_emoji} {direction} — Horizon: {horizon_h}h",
        f"Entry: {_fmt_price(entry_price)}",
    ]

    tp_line = f"Target (touch): {_fmt_price(target_price)}"
    if target_pct is not None:
        tp_line += f" ({target_pct:+.2f}%)"
    lines.append(tp_line)

    sl_line = f"Stop: {_fmt_price(stop_loss)}"
    if stop_pct is not None:
        sl_line += f" ({stop_pct:+.2f}%)"
    lines.append(sl_line)

    if expected_move_pct_f is not None:
        lines.append(f"Expected move (model): {expected_move_pct_f:+.2f}%")

    lines += [
        "",
        f"Model confidence: {confidence:.0%}",
        f"Risk: {risk_label}",
        "Calibrated touch probability:",
        f"• ±1.0% (Analysis): {p1s} ({s5})",
        f"• ±0.5% (Execution): {p05s} ({s6})",
        "",
        f"Win rule: touches target anytime within {horizon_h}h AND correct direction.",
        f"Auto-trade: {auto_status}",
    ]
    if pid_line:
        lines.append(pid_line)

    return "\n".join(lines)


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
    Send alert via Telegram with deduplication, KILLSWITCH, and SMART CAP
    
    KILLSWITCH: If PREDICTIONS_ENABLED=false, ALL alerts are blocked.
    
    SMART CAP RULES (v2.0):
    - Max 10 alerts per day (DAILY_ALERT_CAP)
    - Minimum 80% confidence (MIN_ALERT_CONFIDENCE)
    - Last 3 slots require 85%+ confidence
    - Last slot requires 90%+ confidence
    
    Respects ALERT_STYLE env var: \"simple\" or \"verbose\" (default)

    Returns:
        True if alert was sent successfully
        False if skipped (duplicate, capped, killswitch, or failed)
    """
    # ========================================================================
    # KILLSWITCH CHECK - Emergency stop for all predictions
    # ========================================================================
    try:
        from core.prediction_killswitch import get_killswitch
        killswitch = get_killswitch()
        if not killswitch.can_send_prediction():
            if LOGGER:
                LOGGER.warning(f"⛔ KILLSWITCH blocked: {market}/{symbol} - {killswitch.override_reason}")
            return False
    except Exception as e:
        # If killswitch module fails, block predictions (fail closed)
        if LOGGER:
            LOGGER.error(f"⛔ KILLSWITCH error (blocking): {e}")
        return False
    # Filter out 0% confidence (diagnostic only, not real predictions)
    confidence = prediction.get("confidence", 0)
    if confidence < 0.10:
        if LOGGER:
            LOGGER.info(f"Skipping 0% confidence alert: {market}/{symbol}/{horizon_bucket}")
        return False
    
    # ========================================================================
    # SMART CAP CHECK - Prevents spam, only best predictions get through
    # ========================================================================
    cap_allowed, cap_reason = check_smart_cap(symbol, confidence)
    if not cap_allowed:
        if LOGGER:
            LOGGER.info(f"🛑 SMART CAP blocked: {market}/{symbol} - {cap_reason}")
        return False
    
    # If touch-target gating is present, enforce the 70% gate on calibrated probabilities.
    # Stage5 represents analysis tier (±1.0%). Stage6 represents execution tier (±0.5%).
    # By default, require Stage5 at minimum to send alerts.
    has_gate_fields = any(
        k in prediction
        for k in (
            "stage5_ok",
            "stage6_ok",
            "touch_calibrated_1pct",
            "touch_calibrated_0_5pct",
            "gate",
        )
    )
    if has_gate_fields:
        if not bool(prediction.get("stage5_ok")):
            if LOGGER:
                LOGGER.info(f"Skipping gated alert (<70% Stage5): {market}/{symbol}")
            return False
    else:
        # Fallback to raw confidence threshold
        if confidence < MIN_ALERT_CONFIDENCE:
            if LOGGER:
                LOGGER.info(f"Skipping low confidence alert ({confidence:.0%} < {MIN_ALERT_CONFIDENCE:.0%}): {market}/{symbol}")
            return False
    
    # Check deduplication
    if not should_send_alert(market, symbol, horizon_bucket):
        if LOGGER:
            LOGGER.info(f"Skipping duplicate alert: {market}/{symbol}/{horizon_bucket}")
        return False

    # Prefer touch-target + gating format when provided by the prediction payload.
    has_touch_gate_fields = any(
        k in prediction
        for k in (
            "target_price",
            "stage5_ok",
            "stage6_ok",
            "gate",
            "touch_calibrated_1pct",
            "touch_calibrated_0_5pct",
        )
    )
    if has_touch_gate_fields:
        message = format_touch_target_signal(symbol=symbol, prediction=prediction, price_meta=price_meta)
    elif ALERT_STYLE == "simple":
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
        
        # ========================================================================
        # SMART CAP: Increment daily count AFTER successful send
        # ========================================================================
        if not _increment_daily_count(symbol, confidence):
            if LOGGER:
                LOGGER.warning(f"Daily cap reached after sending {symbol} - future alerts blocked")
        
        if LOGGER:
            daily_count = get_daily_alert_count()
            LOGGER.info(f"✅ Sent alert: {market}/{symbol}/{horizon_bucket} (style={ALERT_STYLE}) [{daily_count}/{DAILY_ALERT_CAP} today]")
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
