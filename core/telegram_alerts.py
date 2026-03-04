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
MIN_ALERT_CONFIDENCE = float(os.getenv("MIN_ALERT_CONFIDENCE", "0.60"))  # 60%: actionable signals only

# Show real accuracy in alerts (not fake "85%")
SHOW_REAL_ACCURACY = os.getenv("SHOW_REAL_ACCURACY", "1") == "1"  # Default: ON


def get_real_accuracy_stats() -> dict:
    """
    Fetch REAL accuracy stats from production database.
    No more hardcoded lies - this shows actual win/loss record.
    
    FIXED Dec 22, 2025: Query ghost_predictions table directly if outcomes empty.
    The 25,691 predictions are in ghost_predictions, not all reconciled yet.
    
    Returns:
        {
            "wins": int,
            "losses": int,
            "accuracy_pct": float,
            "total_verified": int,
            "trend_7d": float,  # 7-day accuracy
            "status": str  # "UNVERIFIED", "BUILDING", "VERIFIED"
        }
    """
    try:
        # Try to get from outcome reconciler's Postgres data
        from core.db_pool import get_sync_connection
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            return {"status": "LEARNING", "wins": 0, "losses": 0, "accuracy_pct": 0, "total_verified": 0}
        
        with get_sync_connection() as conn:
            cursor = conn.cursor()
        
            # First try ghost_prediction_outcomes (reconciled data)
            # NOTE: Status is 'completed' not 'closed'! (Fixed Dec 22, 2025)
            cursor.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE hit_direction = 1) as wins,
                    COUNT(*) FILTER (WHERE hit_direction = 0 AND actual_direction IS NOT NULL) as losses,
                    COUNT(*) as total
                FROM ghost_prediction_outcomes
                WHERE status = 'completed'
            """)
            row = cursor.fetchone()
            wins = row[0] or 0
            losses = row[1] or 0
            total = row[2] or 0
            
            # If outcomes table is empty, try to get count from ghost_predictions
            # This shows we have predictions even if not all reconciled
            if total == 0:
                cursor.execute("""
                    SELECT COUNT(*) as total_predictions
                    FROM ghost_predictions
                    WHERE run_at < EXTRACT(EPOCH FROM NOW()) - 172800  -- older than 48h
                """)
                pred_row = cursor.fetchone()
                total_predictions = pred_row[0] or 0
                
                cursor.close()
                
                # Show "LEARNING" status with prediction count
                return {
                    "wins": 0,
                    "losses": 0,
                    "accuracy_pct": 0,
                    "total_verified": 0,
                    "total_predictions": total_predictions,
                    "status": "LEARNING",
                    "message": f"{total_predictions} predictions pending validation"
                }
        
            # Get 7-day accuracy
            cursor.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE hit_direction = 1) as wins_7d,
                    COUNT(*) as total_7d
                FROM ghost_prediction_outcomes
                WHERE status = 'completed' 
                AND closed_at > NOW() - INTERVAL '7 days'
            """)
            row_7d = cursor.fetchone()
            wins_7d = row_7d[0] or 0
            total_7d = row_7d[1] or 0
            
            cursor.close()
            
            # Calculate accuracy
            accuracy_pct = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            trend_7d = (wins_7d / total_7d * 100) if total_7d > 0 else 0
            
            # Determine status
            if total < 10:
                status = "BUILDING"
            elif accuracy_pct >= 60:
                status = "VERIFIED ✅"
            elif accuracy_pct >= 40:
                status = "MODERATE ⚡"
            else:
                status = "LEARNING 📚"
            
            return {
                "wins": wins,
                "losses": losses,
                "accuracy_pct": accuracy_pct,
                "total_verified": wins + losses,
                "trend_7d": trend_7d,
                "status": status
            }
        
    except Exception as e:
        if LOGGER:
            LOGGER.warning(f"Could not fetch real accuracy: {e}")
        return {"status": "UNVERIFIED", "wins": 0, "losses": 0, "accuracy_pct": 0, "total_verified": 0}

# ============================================================================
# SMART CAP SYSTEM - Prevents spam, only sends best predictions
# ============================================================================
DAILY_ALERT_CAP = int(os.getenv("DAILY_ALERT_CAP", "10"))  # Max 10 alerts per day
SMART_CAP_ENABLED = os.getenv("SMART_CAP_ENABLED", "1") == "1"  # Enable by default

# Thread-safe daily tracking (memory fallback)
_daily_lock = threading.Lock()
_daily_alerts_sent: Dict[str, int] = {}  # {date_str: count} - memory fallback
_daily_alerts_log: Dict[str, List[Dict]] = {}  # {date_str: [{symbol, confidence, timestamp}]}

# Redis keys for persistence (survives restarts!)
REDIS_DAILY_COUNT_KEY = "ghost:alerts:daily_count"
REDIS_DAILY_LOG_KEY = "ghost:alerts:daily_log"


def _get_today_key() -> str:
    """Get today's date key in Chicago timezone"""
    try:
        return datetime.now(ZoneInfo(DEFAULT_TZ)).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def get_daily_alert_count() -> int:
    """Get how many alerts have been sent today (Redis-backed, survives restarts)"""
    today = _get_today_key()
    
    # Try Redis first (persistent across restarts)
    if REDIS_CLIENT:
        try:
            redis_key = f"{REDIS_DAILY_COUNT_KEY}:{today}"
            count = REDIS_CLIENT.get(redis_key)
            if count is not None:
                return int(count)
        except Exception as e:
            if LOGGER:
                LOGGER.warning(f"Redis read failed, using memory: {e}")
    
    # Fallback to memory
    with _daily_lock:
        return _daily_alerts_sent.get(today, 0)


def get_daily_alert_log() -> List[Dict]:
    """Get list of alerts sent today (Redis-backed)"""
    today = _get_today_key()
    
    # Try Redis first
    if REDIS_CLIENT:
        try:
            import json
            log_key = f"{REDIS_DAILY_LOG_KEY}:{today}"
            log_entries = REDIS_CLIENT.lrange(log_key, 0, -1)
            if log_entries:
                return [json.loads(entry) for entry in log_entries]
        except Exception as e:
            if LOGGER:
                LOGGER.warning(f"Redis log read failed: {e}")
    
    # Fallback to memory
    with _daily_lock:
        return _daily_alerts_log.get(today, []).copy()


def _increment_daily_count(symbol: str, confidence: float) -> bool:
    """
    Increment daily alert count (Redis-backed, survives restarts!).
    Returns True if under cap, False if capped.
    """
    today = _get_today_key()
    
    # Try Redis first (persistent across restarts)
    if REDIS_CLIENT:
        try:
            redis_key = f"{REDIS_DAILY_COUNT_KEY}:{today}"
            
            # Atomic increment
            current_count = REDIS_CLIENT.incr(redis_key)
            
            # Set expiry to 48 hours (auto-cleanup)
            REDIS_CLIENT.expire(redis_key, 172800)
            
            # Log to Redis list
            log_key = f"{REDIS_DAILY_LOG_KEY}:{today}"
            import json
            log_entry = json.dumps({
                "symbol": symbol,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "count": current_count
            })
            REDIS_CLIENT.rpush(log_key, log_entry)
            REDIS_CLIENT.expire(log_key, 172800)
            
            if LOGGER:
                LOGGER.info(f"📊 Daily alert count: {current_count}/{DAILY_ALERT_CAP} (Redis-backed)")
            
            return current_count <= DAILY_ALERT_CAP
            
        except Exception as e:
            if LOGGER:
                LOGGER.warning(f"Redis increment failed, using memory: {e}")
    
    # Fallback to memory
    with _daily_lock:
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
    """
    Format prediction as SIMPLE BUY/WAIT alert for buy-low-sell-high traders.
    
    🟢 BUY = Buy now at this price, sell at target
    🔴 WAIT = Don't buy, price dropping, wait for next BUY signal
    """
    # Core fields
    horizon_h = int(prediction.get("horizon_h") or 48)
    confidence = float(prediction.get("confidence") or 0.0)

    action = str(prediction.get("action") or "HOLD").upper()
    direction = str(prediction.get("direction") or "").upper()
    if not direction:
        direction = "UP" if action == "BUY" else "DOWN" if action == "SELL" else "FLAT"

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

    # Format timeframe nicely
    if horizon_h <= 6:
        timeframe = f"{horizon_h} hours"
    elif horizon_h <= 24:
        timeframe = "24 hours"
    elif horizon_h <= 48:
        timeframe = "2 days"
    else:
        timeframe = f"{horizon_h // 24} days"
    
    # Get asset type from classifier
    try:
        from core.asset_classifier import get_asset_type
        asset_type = get_asset_type(symbol)
    except Exception:
        asset_type = "crypto"  # Default
    
    # Get real accuracy stats
    wins, losses, acc_pct = 0, 0, 0
    status_msg = "LEARNING"
    try:
        if SHOW_REAL_ACCURACY:
            stats = get_real_accuracy_stats()
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            acc_pct = stats.get("accuracy_pct", 0)
            status_msg = stats.get("status", "LEARNING")
    except Exception:
        pass
    
    # Format track record - show LEARNING if no verified outcomes yet
    if wins == 0 and losses == 0:
        track_record = f"📊 Status: {status_msg}"
    else:
        track_record = f"📊 Track Record: {wins}W/{losses}L ({acc_pct:.0f}%)"

    # ============================================
    # UPDATED FORMAT Dec 22, 2025
    # Includes: Stop, Confidence, Asset Type, INVERSE label
    # ============================================
    
    if direction == "UP":
        # BUY SIGNAL - Price going up, buy now sell higher
        target_gain_pct = ((target_price - entry_price) / entry_price * 100) if entry_price and target_price else 0
        stop_loss_pct = ((entry_price - stop_loss) / entry_price * 100) if entry_price and stop_loss else 0
        
        lines = [
            f"🟢 **BUY {symbol} NOW**",
            "",
            f"💰 Entry: {_fmt_price(entry_price)}",
            f"🎯 Target: {_fmt_price(target_price)} (+{target_gain_pct:.1f}%)",
            f"🛑 Stop: {_fmt_price(stop_loss)} (-{stop_loss_pct:.1f}%)",
            f"⏱️ Horizon: {timeframe}",
            f"📈 Confidence: {confidence*100:.0f}%",
            f"🏷️ Asset: {asset_type}",
            "",
            track_record,
            f"🔄 INVERSE MODE: Ghost flipped → {direction}"
        ]

    elif direction == "DOWN":
        # WAIT/SHORT SIGNAL - Price going down
        drop_pct = ((entry_price - target_price) / entry_price * 100) if entry_price and target_price else 0
        stop_pct = ((stop_loss - entry_price) / entry_price * 100) if entry_price and stop_loss else 0
        
        lines = [
            f"🔴 **{symbol} — EXPECT DROP**",
            "",
            f"💰 Entry: {_fmt_price(entry_price)}",
            f"🎯 Target: {_fmt_price(target_price)} (-{drop_pct:.1f}%)",
            f"🛑 Stop: {_fmt_price(stop_loss)} (+{stop_pct:.1f}%)",
            f"⏱️ Horizon: {timeframe}",
            f"📈 Confidence: {confidence*100:.0f}%",
            f"🏷️ Asset: {asset_type}",
            "",
            track_record,
            f"🔄 INVERSE MODE: Ghost flipped → {direction}"
        ]
    
    else:
        # FLAT/HOLD - No clear signal
        lines = [
            f"⚪ **{symbol} - No Clear Signal**",
            "",
            f"💰 Current: {_fmt_price(entry_price)}",
            f"⏱️ Horizon: {timeframe}",
            "",
            "_Wait for a clear BUY or WAIT signal_"
        ]

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
    horizon_h: int = 48,
    target_price: float = None,
    stop_loss: float = None,
) -> str:
    """
    Format prediction alert - SIMPLE & CLEAR
    
    For regular traders who BUY LOW, SELL HIGH:
    - BUY signal = Buy now, sell at target
    - WAIT signal = Don't buy, price going down
    """
    conf_pct = int(confidence * 100)
    
    # Get real accuracy stats
    wins, losses, acc_pct = 0, 0, 0
    try:
        if SHOW_REAL_ACCURACY:
            stats = get_real_accuracy_stats()
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            acc_pct = stats.get("accuracy_pct", 0)
    except Exception:
        pass
    
    # Calculate target/stop if not provided
    if target_price is None:
        target_pct = min(abs(change_pct) if change_pct else 3.0, 10.0)  # Cap at 10%
        if direction == "UP":
            target_price = price * (1 + target_pct / 100)
        else:
            target_price = price * (1 - target_pct / 100)
    
    if stop_loss is None:
        stop_pct = min(target_pct * 0.5, 5.0)  # Stop at half target, cap 5%
        if direction == "UP":
            stop_loss = price * (1 - stop_pct / 100)
        else:
            stop_loss = price * (1 + stop_pct / 100)
    
    # Format timeframe nicely
    if horizon_h <= 6:
        timeframe = f"{horizon_h} hours"
    elif horizon_h <= 24:
        timeframe = "24 hours"
    elif horizon_h <= 48:
        timeframe = "2 days"
    else:
        timeframe = f"{horizon_h // 24} days"
    
    # ============================================
    # SIMPLE FORMAT FOR BUY LOW, SELL HIGH TRADERS
    # ============================================
    
    if direction == "UP":
        # BUY SIGNAL - Price going up, buy now sell higher
        target_gain_pct = ((target_price - price) / price) * 100
        stop_loss_pct = ((price - stop_loss) / price) * 100
        
        message = f"""🟢 **BUY {symbol} NOW**

💰 Buy at: ${price:,.0f}
🎯 Sell at: ${target_price:,.0f} (+{target_gain_pct:.1f}%)
🛑 Stop loss: ${stop_loss:,.0f} (-{stop_loss_pct:.1f}%)
⏱️ Timeframe: {timeframe}

📊 Track Record: {wins}W/{losses}L ({acc_pct:.0f}%)"""

    else:
        # WAIT SIGNAL - Price going down, don't buy yet
        message = f"""🔴 **WAIT - Don't Buy {symbol}**

📉 Price expected to DROP
💰 Current: ${price:,.0f}
📍 Wait for: ~${target_price:,.0f} (-{abs(change_pct):.1f}%)
⏱️ Timeframe: {timeframe}

_Ghost will alert you when it's time to BUY_

📊 Track Record: {wins}W/{losses}L ({acc_pct:.0f}%)"""

    return message


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
    
    TOP 10 AGGREGATOR (NEW):
    - Predictions are queued and sent as ONE combined message
    - Individual alerts are DISABLED by default (set INDIVIDUAL_ALERTS_ENABLED=1 to enable)
    
    Respects ALERT_STYLE env var: \"simple\" or \"verbose\" (default)

    Returns:
        True if alert was sent successfully
        False if skipped (duplicate, capped, killswitch, or failed)
    """
    # ========================================================================
    # TOP 10 AGGREGATOR - Combine predictions into ONE message
    # ========================================================================
    # ========================================================================
    # OLD TOP 10 AGGREGATOR - DISABLED
    # This was causing duplicate messages. Now using ghost_notifications.py
    # ========================================================================
    top10_enabled = False  # DISABLED - was os.getenv("TOP10_AGGREGATOR_ENABLED", "1") == "1"
    individual_alerts = os.getenv("INDIVIDUAL_ALERTS_ENABLED", "0") == "1"
    
    if top10_enabled:  # This block will never run now
        try:
            from core.top10_aggregator import intercept_prediction_for_top10
            
            # Create send function wrapper
            def _send_via_telegram(msg: str) -> bool:
                if TELEGRAM_SEND_FUNC and TELEGRAM_CHAT_ID:
                    return TELEGRAM_SEND_FUNC(TELEGRAM_CHAT_ID, msg)
                return False
            
            # Try to add to TOP 10 queue
            queued = intercept_prediction_for_top10(symbol, prediction, price_meta, _send_via_telegram)
            
            if queued:
                if LOGGER:
                    LOGGER.info(f"📋 [TOP 10] Queued {symbol} for combined message")
                
                # If individual alerts disabled, stop here (prediction is queued)
                if not individual_alerts:
                    return True  # Successfully queued (will be sent in combined message)
            
        except Exception as e:
            if LOGGER:
                LOGGER.warning(f"[TOP 10] Aggregator error (falling back to individual alert): {e}")
    
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
