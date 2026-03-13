"""
Ghost Accuracy Autopilot — Auto-Pause When Things Break
════════════════════════════════════════════════════════════════

Three circuit breakers that protect your money:

1. ACCURACY BREAKER: If real accuracy drops below 40% → pause all picks
   Resume when accuracy recovers above 50% (hysteresis prevents flapping)

2. PRICE FEED BREAKER: If both crypto AND stock feeds die → pause picks
   Resume when at least one feed recovers

3. CONFIDENCE BREAKER: Individual predictions below 55% confidence → skip
   Never stored, never sent, never counted

When paused:
  - Telegram gets a single "⚠️ Ghost paused" message
  - Predictions still run internally (to measure recovery)
  - But they're marked as "monitoring only" — not actionable
  - Resume message sent when conditions improve

Created: March 13, 2026
"""

import logging
import os
import threading
import time
from typing import Optional, Tuple

LOGGER = logging.getLogger("ghost.autopilot")

# ── Circuit Breaker Thresholds ────────────────────────────────
ACCURACY_PAUSE_BELOW = 40.0      # Pause all picks if real accuracy < this
ACCURACY_RESUME_ABOVE = 50.0     # Resume when accuracy recovers above this
CONFIDENCE_FLOOR = 0.55          # Individual prediction minimum confidence
FEED_CHECK_INTERVAL = 300        # Check feeds every 5 minutes

# ── State ─────────────────────────────────────────────────────
_lock = threading.Lock()
_is_paused = False
_pause_reason: Optional[str] = None
_pause_time: float = 0.0
_last_accuracy_check: float = 0.0
_last_feed_check: float = 0.0
_telegram_notified_pause = False
_telegram_notified_resume = False

# Accuracy state
_current_real_accuracy: float = 50.0
_current_total_evaluated: int = 0


def _check_accuracy() -> Tuple[bool, float, int]:
    """
    Check real accuracy from PostgreSQL (ALL predictions, no skip exclusion).
    
    Returns:
        (is_acceptable: bool, accuracy_pct: float, total_evaluated: int)
    """
    try:
        from core.db_pool import get_sync_connection
        cutoff_ts = int(time.time()) - (14 * 86400)  # Last 14 days

        with get_sync_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*), SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END)
                FROM ghost_predictions
                WHERE correct IS NOT NULL AND predicted_at > %s
            """, (cutoff_ts,))
            total, correct = cur.fetchone()
            total = total or 0
            correct = correct or 0

        if total < 20:
            return True, 50.0, total  # Not enough data to judge

        accuracy = round(correct / total * 100, 1)
        return accuracy >= ACCURACY_PAUSE_BELOW, accuracy, total

    except Exception as e:
        LOGGER.warning(f"Autopilot accuracy check failed: {e}")
        return True, 50.0, 0  # Don't pause on check failure


def _check_feeds() -> Tuple[bool, str]:
    """
    Check if at least one price feed is responding.
    
    Returns:
        (feeds_ok: bool, detail: str)
    """
    crypto_ok = False
    stock_ok = False

    try:
        from core.crypto.crypto_providers import get_crypto_price_quorum
        result = get_crypto_price_quorum("BTC")
        if result and result.get("price", 0) > 0:
            crypto_ok = True
    except Exception:
        pass

    try:
        from core.providers.turbo_provider import turbo_stock_price
        result = turbo_stock_price("AAPL")
        if result and result.get("price", 0) > 0:
            stock_ok = True
    except Exception:
        pass

    if crypto_ok and stock_ok:
        return True, "both feeds OK"
    elif crypto_ok:
        return True, "crypto OK, stocks down"
    elif stock_ok:
        return True, "stocks OK, crypto down"
    else:
        return False, "ALL feeds down"


def _send_telegram_alert(message: str) -> None:
    """Send a Telegram alert (fire and forget)."""
    try:
        import httpx
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if token and chat_id:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            for cid in chat_id.split(","):
                cid = cid.strip()
                if cid:
                    httpx.post(
                        url,
                        json={"chat_id": cid, "text": message, "disable_web_page_preview": True},
                        timeout=10,
                    )
    except Exception as e:
        LOGGER.warning(f"Autopilot Telegram alert failed: {e}")


def check_and_update() -> dict:
    """
    Run all circuit breaker checks. Call this periodically (every 5 min).
    
    Returns:
        {"paused": bool, "reason": str, "accuracy": float, "feeds": str}
    """
    global _is_paused, _pause_reason, _pause_time
    global _current_real_accuracy, _current_total_evaluated
    global _last_accuracy_check, _last_feed_check
    global _telegram_notified_pause, _telegram_notified_resume

    now = time.time()
    should_pause = False
    reasons = []

    # ── Accuracy check ──
    if now - _last_accuracy_check > 300:
        acc_ok, accuracy, total = _check_accuracy()
        _current_real_accuracy = accuracy
        _current_total_evaluated = total
        _last_accuracy_check = now

        if not acc_ok:
            should_pause = True
            reasons.append(f"accuracy {accuracy}% < {ACCURACY_PAUSE_BELOW}% threshold ({total} evaluated)")

    # ── Feed check ──
    if now - _last_feed_check > FEED_CHECK_INTERVAL:
        feeds_ok, feed_detail = _check_feeds()
        _last_feed_check = now

        if not feeds_ok:
            should_pause = True
            reasons.append(f"price feeds: {feed_detail}")

    with _lock:
        was_paused = _is_paused

        if should_pause and not _is_paused:
            # PAUSE
            _is_paused = True
            _pause_reason = " | ".join(reasons)
            _pause_time = now
            _telegram_notified_resume = False
            LOGGER.warning(f"🛑 AUTOPILOT PAUSE: {_pause_reason}")

            if not _telegram_notified_pause:
                _send_telegram_alert(
                    f"🛑 Ghost Autopilot PAUSED\n"
                    f"──────────────────\n"
                    f"Reason: {_pause_reason}\n"
                    f"Action: Predictions running in monitor-only mode\n"
                    f"Resume: Automatic when conditions improve\n"
                    f"──────────────────\n"
                    f"No picks will be sent until this resolves."
                )
                _telegram_notified_pause = True

        elif not should_pause and _is_paused:
            # Check for resume (with hysteresis)
            if _current_real_accuracy >= ACCURACY_RESUME_ABOVE:
                _is_paused = False
                _pause_reason = None
                _telegram_notified_pause = False
                LOGGER.info(f"✅ AUTOPILOT RESUMED: accuracy {_current_real_accuracy}% > {ACCURACY_RESUME_ABOVE}%")

                if not _telegram_notified_resume:
                    _send_telegram_alert(
                        f"✅ Ghost Autopilot RESUMED\n"
                        f"──────────────────\n"
                        f"Accuracy recovered to {_current_real_accuracy}%\n"
                        f"Predictions are now LIVE again.\n"
                        f"──────────────────"
                    )
                    _telegram_notified_resume = True

    return {
        "paused": _is_paused,
        "reason": _pause_reason,
        "accuracy": _current_real_accuracy,
        "total_evaluated": _current_total_evaluated,
        "pause_duration_s": int(now - _pause_time) if _is_paused else 0,
    }


def is_paused() -> Tuple[bool, Optional[str]]:
    """
    Check if autopilot has paused predictions.
    
    Returns:
        (is_paused: bool, reason: Optional[str])
    """
    with _lock:
        return _is_paused, _pause_reason


def should_skip_prediction(confidence: float) -> Tuple[bool, Optional[str]]:
    """
    Check if an individual prediction should be skipped.
    
    Reasons to skip:
    1. System is paused (accuracy/feed breaker tripped)
    2. Confidence below floor (55%)
    
    Returns:
        (should_skip: bool, reason: Optional[str])
    """
    with _lock:
        if _is_paused:
            return True, f"autopilot paused: {_pause_reason}"

    if confidence < CONFIDENCE_FLOOR:
        return True, f"confidence {confidence:.1%} < {CONFIDENCE_FLOOR:.0%} floor"

    return False, None


def get_status() -> dict:
    """Return autopilot status for the cockpit."""
    with _lock:
        return {
            "paused": _is_paused,
            "reason": _pause_reason,
            "accuracy": _current_real_accuracy,
            "total_evaluated": _current_total_evaluated,
            "confidence_floor": CONFIDENCE_FLOOR,
            "accuracy_pause_below": ACCURACY_PAUSE_BELOW,
            "accuracy_resume_above": ACCURACY_RESUME_ABOVE,
        }
