"""
Ghost Performance Gate — Auto-Kill Bad Symbols
═══════════════════════════════════════════════════════════════

The single source of truth for whether Ghost should trade a symbol.

Decision flow:
  1. Check PostgreSQL for live accuracy over last 14 days
  2. If accuracy < KILL_THRESHOLD (45%) over MIN_TRADES (20+) → KILLED
  3. If accuracy < WARN_THRESHOLD (50%) over MIN_TRADES → WATCHING
  4. Otherwise → ACTIVE

Killed symbols:
  - Removed from prediction pipeline (no new predictions)
  - Removed from Telegram picks (never shown)
  - Moved to "watching" list (still tracked passively)
  - Auto-reinstated if accuracy rises above REINSTATE_THRESHOLD (55%)

This replaces the static edge set with a dynamic, accuracy-driven filter.
The edge set is still the starting pool — this gate REMOVES symbols from it.

Created: March 13, 2026
"""

import logging
import threading
import time
from typing import Dict, FrozenSet, List, Optional, Tuple

LOGGER = logging.getLogger("ghost.performance_gate")

# ── Thresholds ────────────────────────────────────────────────
KILL_THRESHOLD = 45.0       # Below this % → stop trading (killed)
WARN_THRESHOLD = 50.0       # Below this % → reduced size (watching)
REINSTATE_THRESHOLD = 55.0  # Must exceed this to come back from killed
MIN_TRADES = 20             # Minimum evaluated predictions to judge
LOOKBACK_DAYS = 14          # Only count recent performance

# ── Cache ─────────────────────────────────────────────────────
_gate_cache: Dict[str, dict] = {}
_killed_symbols: set = set()
_watching_symbols: set = set()
_cache_lock = threading.Lock()
_last_refresh: float = 0.0
CACHE_TTL = 300  # 5 minutes


def _refresh_gate() -> None:
    """Pull per-symbol accuracy from PostgreSQL and classify."""
    global _last_refresh
    try:
        from core.db_pool import get_sync_connection
        cutoff_ts = int(time.time()) - (LOOKBACK_DAYS * 86400)

        with get_sync_connection() as conn:
            cur = conn.cursor()
            # Count ALL evaluated predictions (no skip exclusion — honest numbers)
            cur.execute("""
                SELECT
                    symbol,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct
                FROM ghost_predictions
                WHERE correct IS NOT NULL
                  AND predicted_at > %s
                GROUP BY symbol
                HAVING COUNT(*) >= %s
                ORDER BY symbol
            """, (cutoff_ts, MIN_TRADES))
            rows = cur.fetchall()

        new_cache: Dict[str, dict] = {}
        new_killed: set = set()
        new_watching: set = set()

        for symbol, total, correct in rows:
            correct = correct or 0
            accuracy = round(correct / total * 100, 1) if total > 0 else 0.0

            if accuracy < KILL_THRESHOLD:
                status = "KILLED"
                reason = f"{accuracy}% over {total} trades (< {KILL_THRESHOLD}% threshold)"
                new_killed.add(symbol)
            elif accuracy < WARN_THRESHOLD:
                status = "WATCHING"
                reason = f"{accuracy}% over {total} trades (< {WARN_THRESHOLD}% threshold)"
                new_watching.add(symbol)
            else:
                status = "ACTIVE"
                reason = None

            new_cache[symbol] = {
                "total": total,
                "correct": correct,
                "accuracy_pct": accuracy,
                "status": status,
                "reason": reason,
            }

        with _cache_lock:
            _gate_cache.clear()
            _gate_cache.update(new_cache)
            _killed_symbols.clear()
            _killed_symbols.update(new_killed)
            _watching_symbols.clear()
            _watching_symbols.update(new_watching)
            _last_refresh = time.time()

        if new_killed:
            LOGGER.warning(
                f"🚫 Performance Gate KILLED {len(new_killed)} symbols: "
                + ", ".join(f"{s} ({new_cache[s]['accuracy_pct']}%)" for s in sorted(new_killed))
            )
        if new_watching:
            LOGGER.info(
                f"👁️ Performance Gate WATCHING {len(new_watching)} symbols: "
                + ", ".join(f"{s} ({new_cache[s]['accuracy_pct']}%)" for s in sorted(new_watching))
            )

    except Exception as e:
        LOGGER.warning(f"Performance Gate refresh failed: {e}")


def _ensure_fresh() -> None:
    """Refresh cache if stale."""
    if time.time() - _last_refresh > CACHE_TTL:
        _refresh_gate()


def is_killed(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a symbol is killed (accuracy too low to trade).
    
    Returns:
        (is_killed: bool, reason: Optional[str])
    """
    _ensure_fresh()
    with _cache_lock:
        entry = _gate_cache.get(symbol.upper())
    if not entry:
        return False, None  # No data yet — allow trading
    if entry["status"] == "KILLED":
        return True, entry["reason"]
    return False, None


def is_watching(symbol: str) -> Tuple[bool, Optional[str]]:
    """Check if a symbol is in watching state (degraded but not killed)."""
    _ensure_fresh()
    with _cache_lock:
        entry = _gate_cache.get(symbol.upper())
    if not entry:
        return False, None
    if entry["status"] == "WATCHING":
        return True, entry["reason"]
    return False, None


def get_killed_symbols() -> List[str]:
    """Return list of currently killed symbols."""
    _ensure_fresh()
    with _cache_lock:
        return sorted(_killed_symbols)


def get_watching_symbols() -> List[str]:
    """Return list of symbols in watching state."""
    _ensure_fresh()
    with _cache_lock:
        return sorted(_watching_symbols)


def get_active_edge_set(base_edge: FrozenSet[str]) -> FrozenSet[str]:
    """
    Filter the base edge set by removing killed symbols.
    
    This is the dynamic replacement for the static edge set.
    Symbols that are killed are removed. Watching symbols remain
    (they still trade, just at reduced confidence).
    
    Args:
        base_edge: The static edge set from config/symbols.py
    
    Returns:
        Filtered edge set with killed symbols removed
    """
    _ensure_fresh()
    with _cache_lock:
        killed = set(_killed_symbols)
    active = frozenset(s for s in base_edge if s not in killed)
    if killed & base_edge:
        removed = killed & base_edge
        LOGGER.info(f"🚫 Performance Gate removed {len(removed)} from edge: {sorted(removed)}")
    return active


def get_scorecard() -> Dict[str, dict]:
    """Return the full performance gate scorecard."""
    _ensure_fresh()
    with _cache_lock:
        return dict(_gate_cache)


def get_summary() -> dict:
    """Return a summary of the performance gate state."""
    _ensure_fresh()
    with _cache_lock:
        total = len(_gate_cache)
        killed = len(_killed_symbols)
        watching = len(_watching_symbols)
        active = total - killed - watching
    return {
        "total_symbols": total,
        "active": active,
        "watching": watching,
        "killed": killed,
        "killed_symbols": get_killed_symbols(),
        "watching_symbols": get_watching_symbols(),
        "thresholds": {
            "kill_below": KILL_THRESHOLD,
            "warn_below": WARN_THRESHOLD,
            "reinstate_above": REINSTATE_THRESHOLD,
            "min_trades": MIN_TRADES,
            "lookback_days": LOOKBACK_DAYS,
        },
    }


def force_refresh() -> Dict[str, dict]:
    """Force a cache refresh and return the scorecard."""
    _refresh_gate()
    return get_scorecard()
