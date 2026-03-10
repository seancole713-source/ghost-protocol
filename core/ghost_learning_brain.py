"""
Ghost Learning Brain — Self-Correcting Prediction Intelligence
═══════════════════════════════════════════════════════════════

Ghost looks at its own scorecard and thinks:
  "I'm 3% on DDOG... what if I flip my predictions?"

This module tracks per-symbol accuracy from PostgreSQL and provides
an auto-inversion signal when Ghost is reliably WRONG on a symbol.
The key insight: a model that is consistently wrong is just as useful
as one that's consistently right — you just flip it.

Self-correcting behavior:
  - Once flipped predictions start being correct, accuracy rises
  - When accuracy crosses back above the threshold, inversion stops
  - This creates a natural feedback loop that converges to correctness

Created: March 10, 2026
"""

import logging
import time
import threading
from typing import Dict, Optional, Tuple

LOGGER = logging.getLogger("ghost.learning_brain")

# ── Configuration ─────────────────────────────────────────────
INVERT_ACCURACY_THRESHOLD = 25.0   # Below this % → invert
MIN_EVALUATED_PREDICTIONS = 10     # Need at least this many to judge
CACHE_TTL_SECONDS = 300            # Refresh from PG every 5 minutes
RECENCY_WINDOW_DAYS = 14           # Only look at last 14 days of predictions

# ── In-memory cache ──────────────────────────────────────────
_symbol_accuracy_cache: Dict[str, dict] = {}
_cache_lock = threading.Lock()
_last_refresh: float = 0.0


def _refresh_cache() -> None:
    """Pull per-symbol accuracy from PostgreSQL into memory."""
    global _last_refresh
    try:
        from core.db_pool import get_sync_connection
        import time as _t

        cutoff_ts = int(_t.time()) - (RECENCY_WINDOW_DAYS * 86400)

        with get_sync_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) as incorrect
                FROM ghost_predictions
                WHERE checked = 1
                  AND eval_version NOT LIKE 'skip%%'
                  AND predicted_at > %s
                GROUP BY symbol
                ORDER BY symbol
            """, (cutoff_ts,))

            rows = cur.fetchall()

        new_cache: Dict[str, dict] = {}
        for symbol, total, correct, incorrect in rows:
            correct = correct or 0
            incorrect = incorrect or 0
            accuracy = (correct / total * 100) if total > 0 else 50.0
            should_invert = total >= MIN_EVALUATED_PREDICTIONS and accuracy < INVERT_ACCURACY_THRESHOLD
            new_cache[symbol] = {
                "total": total,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy_pct": round(accuracy, 1),
                "should_invert": should_invert,
                "reason": (
                    f"accuracy {accuracy:.1f}% over {total} predictions (< {INVERT_ACCURACY_THRESHOLD}%)"
                    if should_invert else None
                ),
            }

        with _cache_lock:
            _symbol_accuracy_cache.clear()
            _symbol_accuracy_cache.update(new_cache)
            _last_refresh = time.time()

        inverted = [s for s, d in new_cache.items() if d["should_invert"]]
        if inverted:
            LOGGER.info(
                f"🧠 Ghost Learning Brain: {len(inverted)} symbols flagged for inversion: "
                + ", ".join(f"{s} ({new_cache[s]['accuracy_pct']}%)" for s in inverted)
            )
        else:
            LOGGER.debug(f"🧠 Ghost Learning Brain: All {len(new_cache)} symbols above {INVERT_ACCURACY_THRESHOLD}% threshold")

    except Exception as e:
        LOGGER.warning(f"🧠 Ghost Learning Brain refresh failed: {e}")


def should_invert(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a symbol's predictions should be inverted.
    
    Returns:
        (should_invert: bool, reason: Optional[str])
    """
    global _last_refresh

    # Refresh cache if stale
    if time.time() - _last_refresh > CACHE_TTL_SECONDS:
        _refresh_cache()

    with _cache_lock:
        entry = _symbol_accuracy_cache.get(symbol)

    if not entry:
        return False, None

    return entry["should_invert"], entry.get("reason")


def apply_inversion(
    symbol: str,
    direction: str,
    target_price: float,
    entry_price: float,
    expected_move_pct: Optional[float] = None,
) -> Tuple[str, float, Optional[float], bool]:
    """
    Apply auto-inversion if the symbol is chronically wrong.
    
    Flips direction AND mirrors the target_price around the entry.
    
    Returns:
        (new_direction, new_target_price, new_expected_move_pct, was_inverted)
    """
    do_invert, reason = should_invert(symbol)
    
    if not do_invert or direction not in ("UP", "DOWN"):
        return direction, target_price, expected_move_pct, False

    # Flip direction
    old_dir = direction
    new_direction = "DOWN" if direction == "UP" else "UP"

    # Mirror target around entry price
    #   If target was 5% BELOW entry (DOWN), make it 5% ABOVE (UP)
    delta = target_price - entry_price
    new_target = entry_price - delta

    # Flip expected_move_pct sign
    new_move_pct = -expected_move_pct if expected_move_pct is not None else None

    LOGGER.info(
        f"[{symbol}] 🧠🔄 GHOST LEARNING BRAIN: {old_dir} → {new_direction} | "
        f"target ${target_price:.2f} → ${new_target:.2f} | "
        f"reason: {reason}"
    )

    return new_direction, new_target, new_move_pct, True


def get_scorecard() -> Dict[str, dict]:
    """Return the full per-symbol accuracy scorecard."""
    if time.time() - _last_refresh > CACHE_TTL_SECONDS:
        _refresh_cache()
    
    with _cache_lock:
        return dict(_symbol_accuracy_cache)


def force_refresh() -> Dict[str, dict]:
    """Force a cache refresh and return the scorecard."""
    _refresh_cache()
    return get_scorecard()
