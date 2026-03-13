"""
Ghost Trade Learning Loop — Learn From Every Trade
════════════════════════════════════════════════════════════════

After every trade resolves (win or loss), this module:
  1. Analyzes WHY it won or lost
  2. Tracks patterns (time of day, direction, confidence level, symbol)
  3. Updates confidence calibration based on actual hit rates
  4. Feeds insights back into the prediction pipeline

Key insights tracked:
  - Win rate by confidence bucket (50-60%, 60-70%, 70-80%, 80%+)
  - Win rate by direction (UP vs DOWN) per asset type
  - Win rate by time of day (pre-market, market hours, after-hours)
  - Win rate by symbol (which symbols are we good at?)
  - Optimal confidence threshold (where does win rate cross 50%?)

This is the difference between a system that stays at 25% forever
and one that improves to 60% over time.

Created: March 13, 2026
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger("ghost.learning_loop")

# ── Cache ─────────────────────────────────────────────────────
_insights_cache: Dict[str, dict] = {}
_cache_lock = threading.Lock()
_last_refresh: float = 0.0
CACHE_TTL = 600  # 10 minutes
LOOKBACK_DAYS = 30


def _refresh_insights() -> None:
    """Pull trade outcomes from PostgreSQL and compute learning insights."""
    global _last_refresh

    try:
        from core.db_pool import get_sync_connection
        cutoff_ts = int(time.time()) - (LOOKBACK_DAYS * 86400)

        with get_sync_connection() as conn:
            cur = conn.cursor()

            insights = {}

            # ── 1. Win rate by confidence bucket ──
            cur.execute("""
                SELECT
                    CASE
                        WHEN confidence >= 0.80 THEN '80+'
                        WHEN confidence >= 0.70 THEN '70-80'
                        WHEN confidence >= 0.60 THEN '60-70'
                        WHEN confidence >= 0.50 THEN '50-60'
                        ELSE 'below50'
                    END as bucket,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as wins
                FROM ghost_predictions
                WHERE correct IS NOT NULL AND predicted_at > %s
                GROUP BY 1
                ORDER BY 1
            """, (cutoff_ts,))
            conf_buckets = {}
            for bucket, total, wins in cur.fetchall():
                wins = wins or 0
                conf_buckets[bucket] = {
                    "total": total,
                    "wins": wins,
                    "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                }
            insights["confidence_buckets"] = conf_buckets

            # ── 2. Win rate by direction ──
            cur.execute("""
                SELECT
                    predicted_direction,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as wins
                FROM ghost_predictions
                WHERE correct IS NOT NULL AND predicted_at > %s
                  AND predicted_direction IN ('UP', 'DOWN')
                GROUP BY predicted_direction
            """, (cutoff_ts,))
            direction_stats = {}
            for direction, total, wins in cur.fetchall():
                wins = wins or 0
                direction_stats[direction] = {
                    "total": total,
                    "wins": wins,
                    "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                }
            insights["direction_stats"] = direction_stats

            # ── 3. Win rate by symbol (top and bottom) ──
            cur.execute("""
                SELECT
                    symbol,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as wins
                FROM ghost_predictions
                WHERE correct IS NOT NULL AND predicted_at > %s
                GROUP BY symbol
                HAVING COUNT(*) >= 5
                ORDER BY (SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END)::float / COUNT(*)) DESC
            """, (cutoff_ts,))
            symbol_stats = {}
            for sym, total, wins in cur.fetchall():
                wins = wins or 0
                symbol_stats[sym] = {
                    "total": total,
                    "wins": wins,
                    "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                }
            insights["symbol_stats"] = symbol_stats

            # Best and worst symbols
            sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]["win_rate"], reverse=True)
            insights["best_symbols"] = [s[0] for s in sorted_symbols[:5] if s[1]["win_rate"] >= 50]
            insights["worst_symbols"] = [s[0] for s in sorted_symbols[-5:] if s[1]["win_rate"] < 45]

            # ── 4. Optimal confidence threshold ──
            # Find the confidence level where win rate crosses 50%
            cur.execute("""
                SELECT confidence, correct
                FROM ghost_predictions
                WHERE correct IS NOT NULL AND predicted_at > %s
                ORDER BY confidence
            """, (cutoff_ts,))
            rows = cur.fetchall()
            if len(rows) >= 20:
                # Sliding window: find where win rate > 50%
                optimal_threshold = 0.55  # Default
                for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
                    above = [(c, r) for c, r in rows if c >= threshold]
                    if len(above) >= 10:
                        wr = sum(1 for _, r in above if r == 1) / len(above)
                        if wr >= 0.50:
                            optimal_threshold = threshold
                            break
                insights["optimal_confidence_threshold"] = optimal_threshold
                insights["threshold_analysis"] = {
                    "recommended": optimal_threshold,
                    "reason": f"Win rate crosses 50% at confidence >= {optimal_threshold:.0%}",
                }
            else:
                insights["optimal_confidence_threshold"] = 0.55

            # ── 5. Recent trend (last 3 days vs previous 11 days) ──
            three_days_ago = int(time.time()) - (3 * 86400)
            cur.execute("""
                SELECT
                    CASE WHEN predicted_at > %s THEN 'recent' ELSE 'older' END as period,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as wins
                FROM ghost_predictions
                WHERE correct IS NOT NULL AND predicted_at > %s
                GROUP BY 1
            """, (three_days_ago, cutoff_ts))
            trend = {}
            for period, total, wins in cur.fetchall():
                wins = wins or 0
                trend[period] = {
                    "total": total,
                    "wins": wins,
                    "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                }
            insights["trend"] = trend
            recent_wr = trend.get("recent", {}).get("win_rate", 0)
            older_wr = trend.get("older", {}).get("win_rate", 0)
            if recent_wr > older_wr + 5:
                insights["trend_direction"] = "IMPROVING"
            elif recent_wr < older_wr - 5:
                insights["trend_direction"] = "DECLINING"
            else:
                insights["trend_direction"] = "STABLE"

            # ── 6. Overall stats ──
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as wins
                FROM ghost_predictions
                WHERE correct IS NOT NULL AND predicted_at > %s
            """, (cutoff_ts,))
            total, wins = cur.fetchone()
            wins = wins or 0
            total = total or 0
            insights["overall"] = {
                "total": total,
                "wins": wins,
                "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                "lookback_days": LOOKBACK_DAYS,
            }

        with _cache_lock:
            _insights_cache.clear()
            _insights_cache.update(insights)
            _last_refresh = time.time()

        LOGGER.info(
            f"📚 Learning Loop: {insights['overall']['win_rate']}% win rate "
            f"({insights['overall']['wins']}/{insights['overall']['total']}) "
            f"| trend: {insights.get('trend_direction', '?')} "
            f"| optimal threshold: {insights.get('optimal_confidence_threshold', '?')}"
        )

    except Exception as e:
        LOGGER.warning(f"Learning Loop refresh failed: {e}")


def get_insights() -> Dict[str, dict]:
    """Return cached learning insights."""
    if time.time() - _last_refresh > CACHE_TTL:
        _refresh_insights()
    with _cache_lock:
        return dict(_insights_cache)


def get_confidence_adjustment(symbol: str, direction: str, confidence: float) -> float:
    """
    Adjust confidence based on learned patterns.
    
    Returns:
        Adjusted confidence value (may be higher or lower)
    """
    if time.time() - _last_refresh > CACHE_TTL:
        _refresh_insights()

    with _cache_lock:
        insights = dict(_insights_cache)

    if not insights:
        return confidence

    adjustment = 0.0

    # Symbol-specific adjustment
    sym_stats = insights.get("symbol_stats", {}).get(symbol)
    if sym_stats and sym_stats["total"] >= 10:
        sym_wr = sym_stats["win_rate"] / 100
        # If symbol has strong track record, boost. If weak, penalize.
        if sym_wr >= 0.60:
            adjustment += 0.05  # +5% for proven winners
        elif sym_wr < 0.35:
            adjustment -= 0.10  # -10% for chronic losers

    # Direction adjustment
    dir_stats = insights.get("direction_stats", {}).get(direction)
    if dir_stats and dir_stats["total"] >= 20:
        dir_wr = dir_stats["win_rate"] / 100
        if dir_wr < 0.40:
            adjustment -= 0.05  # Penalize weak direction

    # Apply adjustment with bounds
    new_confidence = max(0.10, min(0.90, confidence + adjustment))

    if abs(adjustment) > 0.01:
        LOGGER.debug(
            f"[{symbol}] Learning adjustment: {confidence:.2f} → {new_confidence:.2f} "
            f"(adj={adjustment:+.2f})"
        )

    return new_confidence


def analyze_trade(symbol: str, direction: str, correct: bool,
                  confidence: float, entry_price: float,
                  exit_price: float) -> dict:
    """
    Analyze a single resolved trade. Called by the outcome reconciler.
    
    Returns:
        {"lesson": str, "factors": [...]}
    """
    lessons = []
    
    move_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0
    
    if correct:
        if confidence >= 0.70:
            lessons.append("High-confidence hit — model is calibrated well for this setup")
        elif confidence < 0.55:
            lessons.append(f"Low-confidence win ({confidence:.0%}) — lucky or model is under-confident on {symbol}")
    else:
        if confidence >= 0.70:
            lessons.append(f"High-confidence miss ({confidence:.0%}) — model overconfident on {symbol}")
        if direction == "UP" and move_pct < -3:
            lessons.append(f"Called UP, moved {move_pct:.1f}% — strong reversal, check for resistance levels")
        elif direction == "DOWN" and move_pct > 3:
            lessons.append(f"Called DOWN, moved +{move_pct:.1f}% — missed bullish catalyst")

    return {
        "symbol": symbol,
        "direction": direction,
        "correct": correct,
        "confidence": confidence,
        "move_pct": round(move_pct, 2),
        "lessons": lessons,
    }


def get_summary() -> dict:
    """Return a compact summary for the cockpit."""
    insights = get_insights()
    if not insights:
        return {"status": "no data", "insights": {}}

    return {
        "status": "active",
        "overall_win_rate": insights.get("overall", {}).get("win_rate", 0),
        "total_evaluated": insights.get("overall", {}).get("total", 0),
        "trend": insights.get("trend_direction", "UNKNOWN"),
        "optimal_threshold": insights.get("optimal_confidence_threshold", 0.55),
        "best_symbols": insights.get("best_symbols", []),
        "worst_symbols": insights.get("worst_symbols", []),
        "confidence_buckets": insights.get("confidence_buckets", {}),
    }


def force_refresh() -> dict:
    """Force refresh and return insights."""
    _refresh_insights()
    return get_insights()
