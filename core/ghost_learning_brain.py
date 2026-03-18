"""
Ghost Learning Brain — Self-Correcting Prediction Intelligence
═══════════════════════════════════════════════════════════════

Ghost looks at its own scorecard and thinks:
  "I'm 3% on DDOG... what if I flip my predictions?"
  "I'm 29% on XPO... let me bench that one for now."

This module tracks per-symbol accuracy from PostgreSQL and provides:
  1. AUTO-INVERSION — when Ghost is reliably WRONG (<35%), flip it
  2. AUTO-BENCHING — when accuracy is in the dead zone (35-45%),
     bench the symbol from picks entirely until it improves

Three zones:
  ┌─────────────────────────────────────────────────┐
  │  > 45%  │  RECOMMEND  │ Send to Telegram picks  │
  │ 35-45%  │  BENCHED    │ Drop from picks          │
  │  < 35%  │  INVERTED   │ Flip direction           │
  └─────────────────────────────────────────────────┘

Self-correcting behavior:
  - Once flipped predictions start being correct, accuracy rises
  - When accuracy crosses back above the threshold, inversion stops
  - Benched symbols are still tracked — when they improve, they return
  - This creates a natural feedback loop that converges to correctness

Created: March 10, 2026
Updated: March 12, 2026 — Added quality gate (bench losers)
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger("ghost.learning_brain")

# ── Configuration ─────────────────────────────────────────────
BENCH_ACCURACY_THRESHOLD = 45.0    # Below this % → bench (don't recommend)
INVERT_ACCURACY_THRESHOLD = 0.0    # DISABLED — inversions caused feedback loops (Step 6+9)
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
            has_enough_data = total >= MIN_EVALUATED_PREDICTIONS

            # Three zones: INVERT < 35% < BENCH < 45% < RECOMMEND
            should_invert = has_enough_data and accuracy < INVERT_ACCURACY_THRESHOLD
            should_bench = has_enough_data and accuracy < BENCH_ACCURACY_THRESHOLD and not should_invert

            if should_invert:
                status = "INVERTED"
                reason = (
                    f"accuracy {accuracy:.1f}% over {total} preds "
                    f"(< {INVERT_ACCURACY_THRESHOLD}%) → FLIPPING direction"
                )
            elif should_bench:
                status = "BENCHED"
                reason = (
                    f"accuracy {accuracy:.1f}% over {total} preds "
                    f"(< {BENCH_ACCURACY_THRESHOLD}%) → BENCHED from picks"
                )
            else:
                status = "ACTIVE"
                reason = None

            new_cache[symbol] = {
                "total": total,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy_pct": round(accuracy, 1),
                "should_invert": should_invert,
                "should_bench": should_bench,
                "status": status,
                "reason": reason,
            }

        with _cache_lock:
            _symbol_accuracy_cache.clear()
            _symbol_accuracy_cache.update(new_cache)
            _last_refresh = time.time()

        inverted = [s for s, d in new_cache.items() if d["should_invert"]]
        benched = [s for s, d in new_cache.items() if d["should_bench"]]
        if inverted or benched:
            parts = []
            if inverted:
                parts.append(
                    f"{len(inverted)} INVERTED: "
                    + ", ".join(f"{s} ({new_cache[s]['accuracy_pct']}%)" for s in inverted)
                )
            if benched:
                parts.append(
                    f"{len(benched)} BENCHED: "
                    + ", ".join(f"{s} ({new_cache[s]['accuracy_pct']}%)" for s in benched)
                )
            LOGGER.info(f"🧠 Ghost Learning Brain: {' | '.join(parts)}")
        else:
            LOGGER.debug(f"🧠 Ghost Learning Brain: All {len(new_cache)} symbols above {BENCH_ACCURACY_THRESHOLD}% threshold")

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


def should_bench(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a symbol should be BENCHED from picks.
    
    Benched = accuracy is in the dead zone (35-45%).
    Not wrong enough to flip, not right enough to recommend.
    Ghost drops it and moves on to better symbols.
    
    Returns:
        (should_bench: bool, reason: Optional[str])
    """
    global _last_refresh

    # Refresh cache if stale
    if time.time() - _last_refresh > CACHE_TTL_SECONDS:
        _refresh_cache()

    with _cache_lock:
        entry = _symbol_accuracy_cache.get(symbol)

    if not entry:
        return False, None

    # Bench check includes BOTH benched AND inverted symbols
    # Inverted symbols are already flipped at engine level, but if they're
    # still losing after inversion, they should be benched too
    if entry["should_bench"]:
        return True, entry.get("reason")

    return False, None


def is_symbol_blocked(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a symbol should be BLOCKED from Telegram picks.
    This covers BOTH benched (dead zone) AND inverted-but-still-losing symbols.
    
    Use this as the single quality gate for the picks pipeline.
    
    Returns:
        (is_blocked: bool, reason: Optional[str])
    """
    global _last_refresh

    if time.time() - _last_refresh > CACHE_TTL_SECONDS:
        _refresh_cache()

    with _cache_lock:
        entry = _symbol_accuracy_cache.get(symbol)

    if not entry:
        return False, None

    if entry["should_bench"]:
        return True, entry.get("reason")

    return False, None


def get_benched_symbols() -> List[str]:
    """Return list of symbols currently benched from picks."""
    if time.time() - _last_refresh > CACHE_TTL_SECONDS:
        _refresh_cache()

    with _cache_lock:
        return [s for s, d in _symbol_accuracy_cache.items() if d["should_bench"]]


def get_inverted_symbols() -> List[str]:
    """Return list of symbols currently being inverted."""
    if time.time() - _last_refresh > CACHE_TTL_SECONDS:
        _refresh_cache()

    with _cache_lock:
        return [s for s, d in _symbol_accuracy_cache.items() if d["should_invert"]]


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
