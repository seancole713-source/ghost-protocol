"""
Ghost Market Intelligence — Post-Model Enhancement Layer
=========================================================

This module provides market context features that the XGBoost model
doesn't see. These are applied as confidence adjustments AFTER the
model makes its prediction, not as model inputs.

The XGBoost v2 model is pre-trained with 53 fixed features. We can't
change its inputs, but we CAN adjust its outputs based on additional
market intelligence:

1. MOMENTUM ACCELERATION — Is the move speeding up or slowing down?
2. VOLATILITY REGIME — Is ATR unusually high/low vs its 30-day history?
3. VOLUME CONFIRMATION — Does volume support the predicted move?
4. MEAN REVERSION — Is price far from its mean (likely to revert)?
5. CROSS-ASSET CORRELATION — SPY for stocks, BTC for crypto
6. TIME-OF-DAY AWARENESS — Markets behave differently at different hours
7. SYMBOL ACCURACY TRACKING — Per-symbol historical accuracy from PG

Created: Step 11 (Mar 18, 2026) — Phase 2 prediction model upgrade
"""

import logging
import math
import os
import time
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("ghost.market_intelligence")


# ============================================================================
# 1. MOMENTUM ACCELERATION
# ============================================================================

def compute_momentum_acceleration(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute rate-of-change acceleration across timeframes.

    A move that's accelerating (getting faster) is more likely to continue.
    A move that's decelerating is more likely to reverse.

    Returns:
        {
            "acceleration": float (-1 to +1),
            "confidence_adjust": float (-0.05 to +0.08),
            "signal": str ("accelerating_up", "decelerating", "none")
        }
    """
    # Get momentum at different timeframes
    mom_1d = features.get("MOMENTUM_1D", features.get("MOMENTUM_1H", 0)) or 0
    mom_7d = features.get("MOMENTUM_7D", features.get("MOMENTUM_4H", 0)) or 0
    mom_30d = features.get("MOMENTUM_30D", features.get("MOMENTUM_24H", 0)) or 0

    # Price changes across timeframes
    pc_1h = features.get("PRICE_CHANGE_1H", 0) or 0
    pc_4h = features.get("PRICE_CHANGE_4H", 0) or 0
    pc_24h = features.get("PRICE_CHANGE_24H", 0) or 0

    result = {"acceleration": 0.0, "confidence_adjust": 0.0, "signal": "none"}

    # Check if short-term momentum is stronger than long-term (acceleration)
    if abs(mom_1d) > 0.001 and abs(mom_7d) > 0.001:
        # Acceleration = short-term momentum / long-term momentum
        # > 1 means accelerating, < 1 means decelerating
        if mom_1d * mom_7d > 0:  # Same direction
            accel = abs(mom_1d) / max(abs(mom_7d), 0.001)
            if accel > 1.5:
                # Move is accelerating — boost confidence
                result["acceleration"] = min(1.0, accel / 3.0)
                result["confidence_adjust"] = 0.06
                result["signal"] = "accelerating_up" if mom_1d > 0 else "accelerating_down"
            elif accel < 0.5:
                # Move is decelerating — reduce confidence
                result["acceleration"] = -min(1.0, (1.0 / max(accel, 0.01)) / 3.0)
                result["confidence_adjust"] = -0.04
                result["signal"] = "decelerating"
        else:
            # Short and long-term momentum are in different directions
            # This is a potential reversal — strong signal
            result["acceleration"] = -0.5
            result["confidence_adjust"] = -0.05
            result["signal"] = "divergence"

    # Also check price change acceleration
    if abs(pc_1h) > 0.3 and abs(pc_4h) > 0.3:
        if pc_1h * pc_4h > 0 and abs(pc_1h) > abs(pc_4h) * 0.5:
            # Recent move is strong and in same direction — slight boost
            result["confidence_adjust"] = max(result["confidence_adjust"], 0.03)

    return result


# ============================================================================
# 2. VOLATILITY REGIME
# ============================================================================

def compute_volatility_regime(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess whether current volatility is unusually high/low.

    High-volatility environments have larger moves but also more reversals.
    Low-volatility environments can precede breakouts.

    Returns:
        {
            "regime": str ("high_vol", "normal", "low_vol", "compression"),
            "atr_percentile": float (0-1),
            "bb_width_signal": str ("wide", "normal", "compressed"),
            "confidence_adjust": float (-0.05 to +0.05)
        }
    """
    atr = features.get("ATR_14", 0) or 0
    atr_pct = features.get("ATR_PCT", 0) or 0
    bb_width = features.get("BB_WIDTH", 0) or 0
    current_price = features.get("current_price", features.get("price", 0)) or 0
    daily_range = features.get("DAILY_RANGE_PCT", 0) or 0
    vol_7d = features.get("VOLATILITY_7D", features.get("VOLATILITY_20", 0)) or 0
    vol_30d = features.get("VOLATILITY_30D", features.get("VOLATILITY_20D", 0)) or 0

    result = {
        "regime": "normal",
        "atr_percentile": 0.5,
        "bb_width_signal": "normal",
        "confidence_adjust": 0.0,
    }

    # ATR relative assessment
    if current_price > 0 and atr > 0:
        atr_pct_calc = (atr / current_price) * 100
        if atr_pct_calc > 4.0:
            result["regime"] = "high_vol"
            result["atr_percentile"] = 0.85
            # High vol = bigger moves possible but also more noise
            result["confidence_adjust"] = -0.03
        elif atr_pct_calc < 1.0:
            result["regime"] = "low_vol"
            result["atr_percentile"] = 0.15
            # Low vol = potential breakout coming, but current moves are small
            result["confidence_adjust"] = -0.02

    # Bollinger Band width — compression often precedes explosive moves
    if bb_width > 0:
        if bb_width < 0.02:
            result["bb_width_signal"] = "compressed"
            result["regime"] = "compression"
            # Compression = big move coming but DIRECTION is uncertain
            result["confidence_adjust"] = min(result["confidence_adjust"], -0.04)
        elif bb_width > 0.08:
            result["bb_width_signal"] = "wide"
            # Wide bands = high vol, mean reversion more likely
            result["confidence_adjust"] = -0.02

    # Short-term vs long-term volatility divergence
    if vol_7d > 0 and vol_30d > 0:
        vol_ratio = vol_7d / vol_30d
        if vol_ratio > 1.5:
            # Recent vol much higher than average = caution
            result["confidence_adjust"] -= 0.02
        elif vol_ratio < 0.5:
            # Recent vol much lower = potential breakout
            result["regime"] = "compression"

    return result


# ============================================================================
# 3. VOLUME CONFIRMATION
# ============================================================================

def compute_volume_confirmation(
    features: Dict[str, Any], predicted_direction: str
) -> Dict[str, Any]:
    """
    Check if volume confirms the predicted price direction.

    A price move with 2x average volume is more reliable.
    A price move with 0.5x volume is likely to fade.

    Returns:
        {
            "volume_ratio": float,
            "confirmed": bool,
            "confidence_adjust": float (-0.06 to +0.08)
        }
    """
    volume_ratio = features.get("VOLUME_RATIO", 1.0) or 1.0
    volume_spike = features.get("VOLUME_SPIKE", 0) or 0
    obv = features.get("OBV", 0) or 0
    obv_sma = features.get("OBV_SMA", 0) or 0

    result = {
        "volume_ratio": volume_ratio,
        "confirmed": False,
        "confidence_adjust": 0.0,
    }

    # Volume ratio assessment
    if volume_ratio >= 2.0:
        # Strong volume — move is supported
        result["confirmed"] = True
        result["confidence_adjust"] = 0.06
    elif volume_ratio >= 1.5:
        # Above average — moderate support
        result["confirmed"] = True
        result["confidence_adjust"] = 0.03
    elif volume_ratio < 0.5:
        # Very low volume — move is suspect
        result["confirmed"] = False
        result["confidence_adjust"] = -0.05
    elif volume_ratio < 0.7:
        # Below average — mild concern
        result["confidence_adjust"] = -0.02

    # OBV trend check — is money flowing in the predicted direction?
    if obv != 0 and obv_sma != 0:
        obv_trend_up = obv > obv_sma
        if predicted_direction == "UP" and obv_trend_up:
            result["confidence_adjust"] += 0.02  # Money flowing in
        elif predicted_direction == "DOWN" and not obv_trend_up:
            result["confidence_adjust"] += 0.02  # Money flowing out
        elif (predicted_direction == "UP" and not obv_trend_up) or \
             (predicted_direction == "DOWN" and obv_trend_up):
            result["confidence_adjust"] -= 0.03  # Divergence

    return result


# ============================================================================
# 4. MEAN REVERSION SIGNALS
# ============================================================================

def compute_mean_reversion(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess mean reversion potential.

    When price is far from its moving average, it tends to revert.
    This can confirm or contradict the predicted direction.

    Returns:
        {
            "distance_from_mean_pct": float,
            "bb_position": float (0-1),
            "reversion_likely": bool,
            "favored_direction": str ("UP", "DOWN", "NEUTRAL"),
            "confidence_adjust": float (-0.06 to +0.06)
        }
    """
    current_price = features.get("current_price", features.get("price", 0)) or 0
    sma_20 = features.get("SMA_20", features.get("SMA_24", 0)) or 0
    bb_position = features.get("BB_POSITION", 0.5) or 0.5
    rsi = features.get("RSI_14", 50) or 50

    result = {
        "distance_from_mean_pct": 0.0,
        "bb_position": bb_position,
        "reversion_likely": False,
        "favored_direction": "NEUTRAL",
        "confidence_adjust": 0.0,
    }

    # Distance from 20-period moving average
    if current_price > 0 and sma_20 > 0:
        distance_pct = ((current_price - sma_20) / sma_20) * 100
        result["distance_from_mean_pct"] = round(distance_pct, 2)

        if abs(distance_pct) > 5.0:
            # Price is > 5% from its mean — strong reversion signal
            result["reversion_likely"] = True
            result["favored_direction"] = "DOWN" if distance_pct > 0 else "UP"
            result["confidence_adjust"] = 0.05
        elif abs(distance_pct) > 3.0:
            # Moderate deviation
            result["reversion_likely"] = True
            result["favored_direction"] = "DOWN" if distance_pct > 0 else "UP"
            result["confidence_adjust"] = 0.03

    # Bollinger Band extreme positions
    if bb_position <= 0.1:
        # Near lower band — likely to bounce UP
        result["favored_direction"] = "UP"
        result["reversion_likely"] = True
        result["confidence_adjust"] = max(result["confidence_adjust"], 0.04)
    elif bb_position >= 0.9:
        # Near upper band — likely to revert DOWN
        result["favored_direction"] = "DOWN"
        result["reversion_likely"] = True
        result["confidence_adjust"] = max(result["confidence_adjust"], 0.04)

    # RSI extremes reinforce mean reversion
    if rsi < 25 or rsi > 75:
        result["confidence_adjust"] += 0.02

    return result


# ============================================================================
# 5. SPY / MARKET CORRELATION (for stocks)
# ============================================================================

_SPY_CACHE = {"direction": "NEUTRAL", "change_pct": 0.0, "updated": 0.0}
_SPY_LOCK = threading.Lock()


def get_spy_direction() -> Tuple[str, float]:
    """
    Get current SPY direction as a proxy for overall stock market.

    Uses Yahoo Finance (free, no API key).

    Returns:
        (direction: "UP"/"DOWN"/"NEUTRAL", change_pct: float)
    """
    global _SPY_CACHE

    # Cache for 5 minutes
    if time.time() - _SPY_CACHE["updated"] < 300:
        return _SPY_CACHE["direction"], _SPY_CACHE["change_pct"]

    try:
        import yfinance as yf

        ticker = yf.Ticker("SPY")
        hist = ticker.history(period="2d", interval="1h")
        if hist is not None and len(hist) >= 2:
            latest = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-8]) if len(hist) >= 8 else float(hist["Close"].iloc[0])
            change_pct = ((latest - prev) / prev) * 100

            direction = "UP" if change_pct > 0.3 else ("DOWN" if change_pct < -0.3 else "NEUTRAL")

            with _SPY_LOCK:
                _SPY_CACHE["direction"] = direction
                _SPY_CACHE["change_pct"] = change_pct
                _SPY_CACHE["updated"] = time.time()

            return direction, change_pct
    except Exception as e:
        LOGGER.debug(f"SPY direction fetch failed: {e}")

    return _SPY_CACHE["direction"], _SPY_CACHE["change_pct"]


def compute_cross_asset_signal(
    symbol: str, predicted_direction: str, is_crypto: bool
) -> Dict[str, Any]:
    """
    Compute cross-asset correlation signal.

    For crypto: BTC direction (already handled in ensemble_predictor.py)
    For stocks: SPY direction — if SPY is dumping, stock longs are risky

    Returns:
        {
            "reference_asset": str,
            "reference_direction": str,
            "aligned": bool,
            "confidence_adjust": float (-0.05 to +0.05)
        }
    """
    result = {
        "reference_asset": "none",
        "reference_direction": "NEUTRAL",
        "aligned": True,
        "confidence_adjust": 0.0,
    }

    if is_crypto:
        # BTC correlation is already handled in ensemble_predictor.py
        # (get_btc_correlation_boost). Don't double-count.
        result["reference_asset"] = "BTC (handled by ensemble)"
        return result

    # For stocks — use SPY direction
    spy_dir, spy_change = get_spy_direction()
    result["reference_asset"] = "SPY"
    result["reference_direction"] = spy_dir

    if spy_dir == "NEUTRAL":
        return result

    if spy_dir == predicted_direction:
        # Prediction aligns with market direction — slight boost
        result["aligned"] = True
        boost = min(0.05, abs(spy_change) * 0.015)
        result["confidence_adjust"] = boost
    else:
        # Prediction fights the market — moderate penalty
        result["aligned"] = False
        penalty = min(0.05, abs(spy_change) * 0.012)
        result["confidence_adjust"] = -penalty

    return result


# ============================================================================
# 6. TIME-OF-DAY AWARENESS
# ============================================================================

def compute_time_of_day_adjustment(
    symbol: str, is_crypto: bool
) -> Dict[str, Any]:
    """
    Adjust confidence based on time of day and day of week.

    Stock markets:
      - 9:30-10:30 AM ET: Opening volatility — bigger moves, lower reliability
      - 10:30-3:00 PM ET: Normal trading — best reliability
      - 3:00-4:00 PM ET: Power hour — increased volume, moderate reliability
      - 4:00 PM-9:30 AM ET: After/pre-market — low volume, reduce confidence

    Crypto markets:
      - 24/7 trading, but volume is lower on weekends
      - Asian session (8pm-4am ET) often has different dynamics

    Returns:
        {
            "hour_et": int,
            "session": str,
            "is_weekend": bool,
            "confidence_adjust": float (-0.08 to +0.03)
        }
    """
    now_utc = datetime.now(timezone.utc)
    # Convert to US Eastern (UTC-5, or UTC-4 during DST)
    # Simple DST check: March second Sunday to November first Sunday
    # Approximate: if month 3-10, assume EDT (UTC-4), else EST (UTC-5)
    if 3 <= now_utc.month <= 10:
        et_offset = -4
    else:
        et_offset = -5

    from datetime import timedelta
    now_et = now_utc + timedelta(hours=et_offset)
    hour_et = now_et.hour
    minute_et = now_et.minute
    day_of_week = now_et.weekday()  # 0=Mon, 6=Sun
    is_weekend = day_of_week >= 5

    result = {
        "hour_et": hour_et,
        "session": "unknown",
        "is_weekend": is_weekend,
        "day_of_week": day_of_week,
        "confidence_adjust": 0.0,
    }

    if is_crypto:
        # Crypto-specific timing
        if is_weekend:
            result["session"] = "weekend"
            result["confidence_adjust"] = -0.04  # Lower weekend volume
        elif 2 <= hour_et <= 6:
            result["session"] = "low_volume_night"
            result["confidence_adjust"] = -0.03  # Low volume overnight
        elif 9 <= hour_et <= 16:
            result["session"] = "us_overlap"
            result["confidence_adjust"] = 0.02  # Highest crypto volume
        else:
            result["session"] = "normal"
            result["confidence_adjust"] = 0.0
    else:
        # Stock market timing
        if is_weekend:
            result["session"] = "market_closed"
            result["confidence_adjust"] = -0.06  # Market closed
        elif hour_et < 4 or hour_et >= 20:
            result["session"] = "overnight"
            result["confidence_adjust"] = -0.06  # No trading
        elif 4 <= hour_et < 9 or (hour_et == 9 and minute_et < 30):
            result["session"] = "premarket"
            result["confidence_adjust"] = -0.04  # Low volume pre-market
        elif hour_et == 9 and minute_et >= 30 or hour_et == 10 and minute_et < 30:
            result["session"] = "opening"
            result["confidence_adjust"] = -0.02  # Volatile opening
        elif 10 <= hour_et < 15 or (hour_et == 10 and minute_et >= 30):
            result["session"] = "regular"
            result["confidence_adjust"] = 0.03  # Best trading hours
        elif 15 <= hour_et < 16:
            result["session"] = "power_hour"
            result["confidence_adjust"] = 0.01  # High volume but volatile
        elif 16 <= hour_et < 20:
            result["session"] = "after_hours"
            result["confidence_adjust"] = -0.04  # Low volume after hours
        else:
            result["session"] = "normal"
            result["confidence_adjust"] = 0.0

    return result


# ============================================================================
# 7. SYMBOL ACCURACY TRACKING (from PostgreSQL)
# ============================================================================

_SYMBOL_ACCURACY_CACHE: Dict[str, Dict[str, Any]] = {}
_SYMBOL_ACCURACY_LOCK = threading.Lock()
_SYMBOL_ACCURACY_LAST_REFRESH: float = 0.0
_SYMBOL_ACCURACY_TTL = 600  # Refresh every 10 minutes


def refresh_symbol_accuracy_cache() -> None:
    """
    Query PostgreSQL for per-symbol accuracy over the last 14 days.
    Caches results in memory for fast lookup during prediction.
    """
    global _SYMBOL_ACCURACY_LAST_REFRESH

    if time.time() - _SYMBOL_ACCURACY_LAST_REFRESH < _SYMBOL_ACCURACY_TTL:
        return

    try:
        from core.db_pool import get_sync_connection
        import time as _t

        cutoff_ts = int(_t.time()) - (14 * 86400)  # 14 days

        with get_sync_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    symbol,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins,
                    AVG(CASE WHEN correct IS NOT NULL THEN ABS(COALESCE(error_pct, 0)) END) as avg_error
                FROM ghost_predictions
                WHERE checked = 1
                  AND eval_version NOT LIKE 'skip%%'
                  AND predicted_at > %s
                GROUP BY symbol
                HAVING COUNT(*) >= 5
                ORDER BY symbol
            """, (cutoff_ts,))

            rows = cur.fetchall()

        new_cache: Dict[str, Dict[str, Any]] = {}
        for symbol, total, wins, avg_error in rows:
            wins = wins or 0
            accuracy = (wins / total * 100) if total > 0 else 50.0
            new_cache[symbol] = {
                "total": total,
                "wins": wins,
                "accuracy_pct": round(accuracy, 1),
                "avg_error_pct": round(float(avg_error or 0), 2),
            }

        with _SYMBOL_ACCURACY_LOCK:
            _SYMBOL_ACCURACY_CACHE.clear()
            _SYMBOL_ACCURACY_CACHE.update(new_cache)
            _SYMBOL_ACCURACY_LAST_REFRESH = time.time()

        LOGGER.info(
            f"📊 Symbol accuracy cache refreshed: {len(new_cache)} symbols tracked"
        )

    except Exception as e:
        LOGGER.warning(f"Symbol accuracy cache refresh failed: {e}")


def get_symbol_accuracy_adjustment(symbol: str) -> Dict[str, Any]:
    """
    Get confidence adjustment based on per-symbol historical accuracy.

    - Symbols with >60% accuracy get a small confidence boost
    - Symbols with <30% accuracy are blocked (HOLD forced)
    - Symbols with 30-45% accuracy get a confidence penalty

    Returns:
        {
            "symbol_accuracy_pct": float,
            "sample_size": int,
            "confidence_adjust": float (-0.10 to +0.05),
            "should_block": bool,
            "reason": str
        }
    """
    # Refresh cache if stale
    refresh_symbol_accuracy_cache()

    with _SYMBOL_ACCURACY_LOCK:
        entry = _SYMBOL_ACCURACY_CACHE.get(symbol)

    result = {
        "symbol_accuracy_pct": None,
        "sample_size": 0,
        "confidence_adjust": 0.0,
        "should_block": False,
        "reason": "no_data",
    }

    if not entry or entry["total"] < 5:
        result["reason"] = "insufficient_data"
        return result

    accuracy = entry["accuracy_pct"]
    total = entry["total"]
    result["symbol_accuracy_pct"] = accuracy
    result["sample_size"] = total

    if accuracy < 25.0 and total >= 10:
        # Very low accuracy — block this symbol
        result["should_block"] = True
        result["confidence_adjust"] = -0.15
        result["reason"] = f"blocked: {accuracy:.0f}% over {total} predictions"
    elif accuracy < 35.0 and total >= 8:
        # Low accuracy — heavy penalty
        result["confidence_adjust"] = -0.10
        result["reason"] = f"low_accuracy: {accuracy:.0f}% over {total}"
    elif accuracy < 45.0:
        # Below average — moderate penalty
        result["confidence_adjust"] = -0.05
        result["reason"] = f"below_avg: {accuracy:.0f}% over {total}"
    elif accuracy > 65.0 and total >= 10:
        # High accuracy — boost
        result["confidence_adjust"] = 0.05
        result["reason"] = f"high_accuracy: {accuracy:.0f}% over {total}"
    elif accuracy > 55.0 and total >= 8:
        # Above average — small boost
        result["confidence_adjust"] = 0.03
        result["reason"] = f"above_avg: {accuracy:.0f}% over {total}"
    else:
        result["reason"] = f"normal: {accuracy:.0f}% over {total}"

    return result


# ============================================================================
# 8. HISTORICAL CONFIDENCE CALIBRATION (from PostgreSQL)
# ============================================================================

_CONFIDENCE_MAP_CACHE: Dict[int, float] = {}
_CONFIDENCE_MAP_LOCK = threading.Lock()
_CONFIDENCE_MAP_LAST_REFRESH: float = 0.0
_CONFIDENCE_MAP_TTL = 1800  # Refresh every 30 minutes


def refresh_confidence_map() -> None:
    """
    Build a map from raw confidence → actual accuracy rate.

    Example output: {60: 0.38, 65: 0.42, 70: 0.51, 75: 0.55}
    Meaning: When the model says 70% confidence, it's actually right 51%.
    """
    global _CONFIDENCE_MAP_LAST_REFRESH

    if time.time() - _CONFIDENCE_MAP_LAST_REFRESH < _CONFIDENCE_MAP_TTL:
        return

    try:
        from core.db_pool import get_sync_connection
        import time as _t

        cutoff_ts = int(_t.time()) - (30 * 86400)  # 30 days

        with get_sync_connection() as conn:
            cur = conn.cursor()
            # Bin confidence into 5% buckets and compute actual accuracy per bucket
            cur.execute("""
                SELECT
                    FLOOR(confidence * 20) * 5 AS conf_bucket,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins
                FROM ghost_predictions
                WHERE checked = 1
                  AND confidence IS NOT NULL
                  AND eval_version NOT LIKE 'skip%%'
                  AND predicted_at > %s
                GROUP BY FLOOR(confidence * 20) * 5
                HAVING COUNT(*) >= 5
                ORDER BY conf_bucket
            """, (cutoff_ts,))

            rows = cur.fetchall()

        new_map: Dict[int, float] = {}
        for bucket, total, wins in rows:
            wins = wins or 0
            actual_accuracy = wins / total if total > 0 else 0.5
            new_map[int(bucket)] = round(actual_accuracy, 3)

        with _CONFIDENCE_MAP_LOCK:
            _CONFIDENCE_MAP_CACHE.clear()
            _CONFIDENCE_MAP_CACHE.update(new_map)
            _CONFIDENCE_MAP_LAST_REFRESH = time.time()

        if new_map:
            LOGGER.info(
                f"📊 Confidence calibration map: "
                + ", ".join(f"{k}%→{v:.0%}" for k, v in sorted(new_map.items()))
            )
    except Exception as e:
        LOGGER.debug(f"Confidence map refresh failed: {e}")


def calibrate_raw_confidence(raw_confidence: float) -> Dict[str, Any]:
    """
    Map the model's raw confidence to historically-calibrated confidence.

    If the model says 70% but historical accuracy at 70% is only 45%,
    the calibrated confidence becomes 0.45.

    Returns:
        {
            "raw_confidence": float,
            "calibrated_confidence": float,
            "confidence_adjust": float,
            "calibration_source": str
        }
    """
    refresh_confidence_map()

    result = {
        "raw_confidence": raw_confidence,
        "calibrated_confidence": raw_confidence,
        "confidence_adjust": 0.0,
        "calibration_source": "none",
    }

    with _CONFIDENCE_MAP_LOCK:
        if not _CONFIDENCE_MAP_CACHE:
            return result

        # Find the nearest confidence bucket
        raw_pct = int(raw_confidence * 100)
        bucket = (raw_pct // 5) * 5  # Round down to nearest 5

        if bucket in _CONFIDENCE_MAP_CACHE:
            actual = _CONFIDENCE_MAP_CACHE[bucket]
            # Blend: 70% calibrated, 30% raw (don't fully replace model's judgment)
            blended = actual * 0.7 + raw_confidence * 0.3
            result["calibrated_confidence"] = round(blended, 3)
            result["confidence_adjust"] = round(blended - raw_confidence, 3)
            result["calibration_source"] = f"bucket_{bucket}"
        else:
            # Interpolate between nearest buckets
            buckets = sorted(_CONFIDENCE_MAP_CACHE.keys())
            if buckets:
                lower = max((b for b in buckets if b <= raw_pct), default=buckets[0])
                upper = min((b for b in buckets if b >= raw_pct), default=buckets[-1])
                if lower == upper:
                    actual = _CONFIDENCE_MAP_CACHE[lower]
                else:
                    w = (raw_pct - lower) / max(upper - lower, 1)
                    actual = _CONFIDENCE_MAP_CACHE[lower] * (1 - w) + _CONFIDENCE_MAP_CACHE[upper] * w
                blended = actual * 0.7 + raw_confidence * 0.3
                result["calibrated_confidence"] = round(blended, 3)
                result["confidence_adjust"] = round(blended - raw_confidence, 3)
                result["calibration_source"] = f"interpolated_{lower}_{upper}"

    return result


# ============================================================================
# MASTER: Apply all intelligence adjustments
# ============================================================================

def apply_market_intelligence(
    features: Dict[str, Any],
    predicted_direction: str,
    raw_confidence: float,
    symbol: str,
    is_crypto: bool,
) -> Dict[str, Any]:
    """
    Apply all market intelligence adjustments to a prediction.

    This is the single entry point that the ensemble predictor calls.

    Returns:
        {
            "final_confidence": float,
            "final_direction": str,
            "adjustments": dict (each sub-signal's contribution),
            "blocked": bool,
            "block_reason": str | None,
            "intelligence_version": str
        }
    """
    adjustments = {}
    total_adjust = 0.0
    direction = predicted_direction
    blocked = False
    block_reason = None

    # 1. Historical confidence calibration
    cal = calibrate_raw_confidence(raw_confidence)
    if cal["calibration_source"] != "none":
        adjustments["calibration"] = cal["confidence_adjust"]
        total_adjust += cal["confidence_adjust"]
        # Start from calibrated confidence instead of raw
        raw_confidence = cal["calibrated_confidence"]

    # 2. Symbol-specific accuracy
    sym_acc = get_symbol_accuracy_adjustment(symbol)
    adjustments["symbol_accuracy"] = sym_acc["confidence_adjust"]
    total_adjust += sym_acc["confidence_adjust"]
    if sym_acc["should_block"]:
        blocked = True
        block_reason = sym_acc["reason"]

    # 3. Momentum acceleration
    mom = compute_momentum_acceleration(features)
    adjustments["momentum"] = mom["confidence_adjust"]
    total_adjust += mom["confidence_adjust"]

    # 4. Volatility regime
    vol = compute_volatility_regime(features)
    adjustments["volatility"] = vol["confidence_adjust"]
    total_adjust += vol["confidence_adjust"]

    # 5. Volume confirmation
    vol_conf = compute_volume_confirmation(features, predicted_direction)
    adjustments["volume"] = vol_conf["confidence_adjust"]
    total_adjust += vol_conf["confidence_adjust"]

    # 6. Mean reversion
    mr = compute_mean_reversion(features)
    adjustments["mean_reversion"] = mr["confidence_adjust"]
    # Mean reversion adjusts confidence IN THE FAVORED DIRECTION
    # If mean reversion favors the predicted direction, boost; if against, penalize
    if mr["reversion_likely"] and mr["favored_direction"] != "NEUTRAL":
        if mr["favored_direction"] == predicted_direction:
            total_adjust += mr["confidence_adjust"]  # Boost
        else:
            total_adjust -= mr["confidence_adjust"]  # Penalize (opposite direction)
            adjustments["mean_reversion"] = -mr["confidence_adjust"]

    # 7. Cross-asset correlation (SPY for stocks, BTC already in ensemble)
    cross = compute_cross_asset_signal(symbol, predicted_direction, is_crypto)
    adjustments["cross_asset"] = cross["confidence_adjust"]
    total_adjust += cross["confidence_adjust"]

    # 8. Time-of-day awareness
    tod = compute_time_of_day_adjustment(symbol, is_crypto)
    adjustments["time_of_day"] = tod["confidence_adjust"]
    total_adjust += tod["confidence_adjust"]

    # Apply total adjustment, clamped to reasonable range
    # Cap total adjustment to ±20% to prevent wild swings
    total_adjust = max(-0.20, min(0.20, total_adjust))
    final_confidence = max(0.20, min(0.85, raw_confidence + total_adjust))

    # If blocked, force HOLD with very low confidence
    if blocked:
        direction = "HOLD"
        final_confidence = 0.25

    # Log significant adjustments
    if abs(total_adjust) > 0.03:
        active = {k: v for k, v in adjustments.items() if abs(v) > 0.005}
        LOGGER.info(
            f"🧠 Market Intelligence [{symbol}]: {raw_confidence:.0%} → {final_confidence:.0%} "
            f"(Δ{total_adjust:+.0%}) | {active}"
        )

    return {
        "final_confidence": round(final_confidence, 3),
        "final_direction": direction,
        "raw_confidence": round(raw_confidence, 3),
        "total_adjustment": round(total_adjust, 3),
        "adjustments": {k: round(v, 3) for k, v in adjustments.items()},
        "blocked": blocked,
        "block_reason": block_reason,
        "signals": {
            "momentum": mom["signal"],
            "volatility_regime": vol["regime"],
            "volume_confirmed": vol_conf["confirmed"],
            "mean_reversion": mr["reversion_likely"],
            "mr_direction": mr["favored_direction"],
            "session": tod["session"],
            "symbol_accuracy": sym_acc.get("symbol_accuracy_pct"),
        },
        "intelligence_version": "v1",
    }
