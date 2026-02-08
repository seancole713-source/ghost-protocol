"""
Directional Accuracy Tracker — Adaptive UP/DOWN Confidence Adjustment

Tracks UP vs DOWN prediction accuracy from paper_trades in real time
and auto-adjusts confidence penalties/bonuses to reflect actual performance.

This replaces hardcoded penalties with LIVE DATA-DRIVEN adjustments.

Market regime changes (bull → bear → bull) will automatically shift
which direction gets penalized and which gets boosted.

Algorithm:
  1. Query last 14 days of resolved paper trades by direction
  2. Calculate UP WR% and DOWN WR% separately
  3. Compute penalty/bonus relative to 50% baseline:
     - If UP WR = 30% → penalty = (50-30)/100 * scale = -0.12
     - If DOWN WR = 85% → bonus = (85-50)/100 * scale = +0.05 (capped)
     - If UP WR = 65% → bonus = (65-50)/100 * scale = +0.02
  4. Cache for 1 hour to avoid DB spam
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta

LOGGER = logging.getLogger("ghost.directional_tracker")

# Cache for directional accuracy data
_CACHE = {
    "up_wr": None,
    "down_wr": None,
    "up_resolved": 0,
    "down_resolved": 0,
    "up_penalty": 0.12,  # Default until first query
    "down_bonus": 0.05,  # Default until first query
    "last_updated": 0,
    "regime": "unknown",
}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_S = int(os.getenv("DIRECTIONAL_CACHE_TTL_S", "3600"))  # 1 hour default

# How aggressively to penalize/boost based on WR deviation from 50%
# Scale of 0.6 means: 20% WR deviation → 12% confidence adjustment
_PENALTY_SCALE = float(os.getenv("DIRECTIONAL_PENALTY_SCALE", "0.6"))
_MAX_PENALTY = float(os.getenv("DIRECTIONAL_MAX_PENALTY", "0.20"))  # Cap at 20%
_MAX_BONUS = float(os.getenv("DIRECTIONAL_MAX_BONUS", "0.08"))  # Cap at 8%
_MIN_SAMPLE_SIZE = int(os.getenv("DIRECTIONAL_MIN_SAMPLES", "20"))  # Need 20+ trades
_LOOKBACK_DAYS = int(os.getenv("DIRECTIONAL_LOOKBACK_DAYS", "14"))


def _query_directional_accuracy() -> dict:
    """Query paper_trades for UP vs DOWN win rates over the lookback window."""
    try:
        from core.paper_tracker import get_paper_tracker
        tracker = get_paper_tracker()
        conn = tracker._get_connection()

        cutoff = (datetime.utcnow() - timedelta(days=_LOOKBACK_DAYS)).isoformat()

        cur = tracker._execute(conn, """
            SELECT
                signal_direction,
                COUNT(*) as resolved,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome IN ('LOSS', 'STOPPED') THEN 1 ELSE 0 END) as losses
            FROM paper_trades
            WHERE outcome IN ('WIN', 'LOSS', 'STOPPED')
              AND created_at >= ?
            GROUP BY signal_direction
        """, (cutoff,))

        rows = tracker._fetchall(cur)
        conn.close()

        result = {"UP": {"resolved": 0, "wins": 0}, "DOWN": {"resolved": 0, "wins": 0}}
        for row in rows:
            direction = row.get("signal_direction", "").upper()
            if direction in ("UP", "LONG"):
                result["UP"]["resolved"] += row.get("resolved", 0) or 0
                result["UP"]["wins"] += row.get("wins", 0) or 0
            elif direction in ("DOWN", "SHORT"):
                result["DOWN"]["resolved"] += row.get("resolved", 0) or 0
                result["DOWN"]["wins"] += row.get("wins", 0) or 0

        return result
    except Exception as e:
        LOGGER.warning(f"[DIRECTIONAL] Failed to query accuracy: {e}")
        return None


def _calculate_adjustment(win_rate: float, resolved: int) -> float:
    """
    Calculate confidence adjustment based on win rate deviation from 50%.

    Returns negative (penalty) if WR < 50%, positive (bonus) if WR > 50%.
    """
    if resolved < _MIN_SAMPLE_SIZE:
        return 0.0  # Not enough data

    deviation = (win_rate - 0.50)  # e.g., 30% WR → -0.20, 85% WR → +0.35

    adjustment = deviation * _PENALTY_SCALE

    # Apply caps
    if adjustment < 0:
        return max(adjustment, -_MAX_PENALTY)
    else:
        return min(adjustment, _MAX_BONUS)


def _determine_regime(up_wr: float, down_wr: float, up_n: int, down_n: int) -> str:
    """Classify the current market regime based on directional accuracy."""
    if up_n < _MIN_SAMPLE_SIZE and down_n < _MIN_SAMPLE_SIZE:
        return "insufficient_data"

    if up_n >= _MIN_SAMPLE_SIZE and down_n >= _MIN_SAMPLE_SIZE:
        if up_wr >= 0.55 and down_wr >= 0.55:
            return "model_accurate"  # Model is good at both — rare
        elif up_wr < 0.40 and down_wr >= 0.60:
            return "bearish_edge"  # Model good at DOWN, bad at UP
        elif down_wr < 0.40 and up_wr >= 0.60:
            return "bullish_edge"  # Model good at UP, bad at DOWN
        elif up_wr < 0.40 and down_wr < 0.40:
            return "model_broken"  # Model bad at both — big problem
        else:
            return "mixed"

    # Only one direction has enough data
    if up_n >= _MIN_SAMPLE_SIZE:
        return "bullish_edge" if up_wr >= 0.55 else "bearish_edge"
    return "bearish_edge" if down_wr >= 0.55 else "bullish_edge"


def refresh_cache():
    """Refresh the directional accuracy cache from paper_trades."""
    data = _query_directional_accuracy()
    if data is None:
        return

    up_resolved = data["UP"]["resolved"]
    up_wins = data["UP"]["wins"]
    down_resolved = data["DOWN"]["resolved"]
    down_wins = data["DOWN"]["wins"]

    up_wr = up_wins / up_resolved if up_resolved > 0 else None
    down_wr = down_wins / down_resolved if down_resolved > 0 else None

    # Calculate adjustments
    up_adj = _calculate_adjustment(up_wr, up_resolved) if up_wr is not None else -0.12
    down_adj = _calculate_adjustment(down_wr, down_resolved) if down_wr is not None else 0.05

    # Penalties are negative adjustments, bonuses are positive
    up_penalty = abs(up_adj) if up_adj < 0 else 0.0
    up_bonus = up_adj if up_adj > 0 else 0.0
    down_penalty = abs(down_adj) if down_adj < 0 else 0.0
    down_bonus = down_adj if down_adj > 0 else 0.0

    regime = _determine_regime(
        up_wr or 0, down_wr or 0, up_resolved, down_resolved
    )

    with _CACHE_LOCK:
        _CACHE["up_wr"] = up_wr
        _CACHE["down_wr"] = down_wr
        _CACHE["up_resolved"] = up_resolved
        _CACHE["down_resolved"] = down_resolved
        _CACHE["up_penalty"] = up_penalty
        _CACHE["up_bonus"] = up_bonus
        _CACHE["down_penalty"] = down_penalty
        _CACHE["down_bonus"] = down_bonus
        _CACHE["last_updated"] = time.time()
        _CACHE["regime"] = regime

    LOGGER.info(
        f"[DIRECTIONAL] 🔄 Cache refreshed: "
        f"UP={up_wr:.1%} ({up_resolved} trades, adj={up_adj:+.1%}) | "
        f"DOWN={down_wr:.1%} ({down_resolved} trades, adj={down_adj:+.1%}) | "
        f"Regime={regime}"
        if up_wr is not None and down_wr is not None else
        f"[DIRECTIONAL] 🔄 Cache refreshed: insufficient data"
    )


def get_directional_adjustment(direction: str) -> tuple[float, dict]:
    """
    Get the confidence adjustment for a given prediction direction.

    Returns:
        (adjustment, metadata) where adjustment is a float to ADD to confidence
        (negative = penalty, positive = bonus), and metadata contains debug info.
    """
    # Check if cache needs refresh
    with _CACHE_LOCK:
        cache_age = time.time() - _CACHE["last_updated"]
        needs_refresh = cache_age > _CACHE_TTL_S or _CACHE["last_updated"] == 0

    if needs_refresh:
        try:
            refresh_cache()
        except Exception as e:
            LOGGER.warning(f"[DIRECTIONAL] Cache refresh failed: {e}")

    with _CACHE_LOCK:
        direction_upper = direction.upper()

        if direction_upper in ("UP", "LONG"):
            wr = _CACHE["up_wr"]
            resolved = _CACHE["up_resolved"]
            penalty = _CACHE["up_penalty"]
            bonus = _CACHE.get("up_bonus", 0)
            adjustment = bonus - penalty  # net adjustment (negative if penalty > bonus)
        elif direction_upper in ("DOWN", "SHORT"):
            wr = _CACHE["down_wr"]
            resolved = _CACHE["down_resolved"]
            penalty = _CACHE.get("down_penalty", 0)
            bonus = _CACHE["down_bonus"]
            adjustment = bonus - penalty
        else:
            return 0.0, {"reason": f"unknown_direction_{direction}"}

        metadata = {
            "direction": direction_upper,
            "win_rate": round(wr * 100, 1) if wr is not None else None,
            "sample_size": resolved,
            "adjustment": round(adjustment, 4),
            "regime": _CACHE["regime"],
            "lookback_days": _LOOKBACK_DAYS,
            "cache_age_s": int(time.time() - _CACHE["last_updated"]),
            "adaptive": True,
        }

        return adjustment, metadata


def get_regime_info() -> dict:
    """Get current regime info for debugging/monitoring endpoints."""
    with _CACHE_LOCK:
        return {
            "regime": _CACHE["regime"],
            "up_win_rate": round(_CACHE["up_wr"] * 100, 1) if _CACHE["up_wr"] is not None else None,
            "down_win_rate": round(_CACHE["down_wr"] * 100, 1) if _CACHE["down_wr"] is not None else None,
            "up_resolved": _CACHE["up_resolved"],
            "down_resolved": _CACHE["down_resolved"],
            "up_penalty": round(_CACHE["up_penalty"], 4),
            "up_bonus": round(_CACHE.get("up_bonus", 0), 4),
            "down_penalty": round(_CACHE.get("down_penalty", 0), 4),
            "down_bonus": round(_CACHE["down_bonus"], 4),
            "lookback_days": _LOOKBACK_DAYS,
            "min_sample_size": _MIN_SAMPLE_SIZE,
            "penalty_scale": _PENALTY_SCALE,
            "cache_age_s": int(time.time() - _CACHE["last_updated"]),
            "last_updated": datetime.utcfromtimestamp(_CACHE["last_updated"]).isoformat() if _CACHE["last_updated"] else None,
        }
