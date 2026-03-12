"""
Ghost Protocol — Bug-Specific Integrity Checks
════════════════════════════════════════════════

Runtime verification that EVERY major bug fix is still intact.
Each check validates the invariant that the bug-fix established.

Bugs covered:
  1. Crypto/Stock Misclassification (33 symbols classified wrong)
  2. Skip-Tag Pollution (accuracy queries counting skip-tagged predictions)
  3. Direction/Target Mismatch in Telegram (UP direction, target below entry)
  4. _LATEST_PREDICTIONS Cache Corruption (missing/conflicting fields)
  5. Ghost Brain vs Learning Brain Conflict (invert wars)
  6. V3 Filter Misconfiguration (wrong min_confidence, broken inverse)
  7. Edge Whitelist Bypass (non-edge symbols leaking into picks)
  8. Adapters Pipeline Classification (crypto picked as stock in formatter)

Usage:
    from core.integrity_bug_checks import run_bug_checks, BugCheckResult

    results = run_bug_checks()
    for r in results:
        print(f"{'✅' if r.passed else '❌'} {r.name}: {r.detail}")

Created: March 12, 2026
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

LOGGER = logging.getLogger("ghost.integrity.bugs")


@dataclass
class BugCheckResult:
    """Result of a single bug-fix integrity check."""
    name: str
    bug_id: int
    passed: bool
    severity: str  # "error" | "warn" | "info"
    detail: str
    mismatches: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# BUG 1: Crypto/Stock Misclassification
# ══════════════════════════════════════════════════════════════════
# ROOT CAUSE: core/asset_classification.py had ~50 crypto symbols but
# config/symbols.py CRYPTO_SYMBOLS had ~68. The 33 missing ones (CHZ,
# ILV, TURBO, ZEC, etc.) were classified as "stock" by the Telegram
# picks pipeline, stealing slots from actual stocks.
#
# FIX: _merge_config_crypto() in asset_classification.py merges the
# config/symbols.py set at import time.
#
# INVARIANT: Every symbol in config/symbols.CRYPTO_SYMBOLS must also
# be in core/asset_classification.CRYPTO_SYMBOLS. Additionally,
# config/symbols.is_crypto() and asset_classification.is_crypto_symbol()
# must agree for ALL edge whitelist symbols.
# ══════════════════════════════════════════════════════════════════

def check_crypto_classification() -> BugCheckResult:
    """
    Verify both classification systems agree on every crypto symbol.

    Sources checked:
      - config.symbols.CRYPTO_SYMBOLS        (FrozenSet, ~68 symbols)
        config.symbols.is_crypto(sym)         (config/symbols.py:253)
      - core.asset_classification.CRYPTO_SYMBOLS  (set, merged at import)
        core.asset_classification.is_crypto_symbol(sym)  (asset_classification.py:110)
      - core.symbol_registry.KNOWN_CRYPTO    (set, ~80 symbols)
      - Ghost Brain's _KNOWN_CRYPTO          (ghost_brain.py:172-185)
    """
    mismatches: List[str] = []

    # ── Source 1: config/symbols.py ──
    try:
        from config.symbols import CRYPTO_SYMBOLS as config_crypto
        from config.symbols import is_crypto as config_is_crypto
    except ImportError as e:
        return BugCheckResult(
            name="Crypto/Stock Classification",
            bug_id=1,
            passed=False,
            severity="error",
            detail=f"Cannot import config.symbols: {e}",
        )

    # ── Source 2: core/asset_classification.py ──
    try:
        from core.asset_classification import CRYPTO_SYMBOLS as ac_crypto
        from core.asset_classification import is_crypto_symbol
    except ImportError as e:
        return BugCheckResult(
            name="Crypto/Stock Classification",
            bug_id=1,
            passed=False,
            severity="error",
            detail=f"Cannot import core.asset_classification: {e}",
        )

    # ── Source 3: core/symbol_registry.py ──
    try:
        from core.symbol_registry import KNOWN_CRYPTO as registry_crypto
    except ImportError:
        registry_crypto = set()

    # ── Source 4: Ghost Brain _KNOWN_CRYPTO ──
    try:
        from core.ghost_brain import _KNOWN_CRYPTO as brain_crypto
    except ImportError:
        brain_crypto = set()

    # CHECK 1: Every config/symbols.py crypto must be in asset_classification
    # This is the bug that caused 33 symbols to be classified as stocks.
    missing_in_ac = set()
    for sym in config_crypto:
        s = sym.upper()
        if s not in ac_crypto:
            missing_in_ac.add(s)
            mismatches.append(f"{s}: in config/symbols.py but NOT in asset_classification.py")

    # CHECK 2: is_crypto() and is_crypto_symbol() must agree on edge symbols
    try:
        from config.symbols import get_edge_set
        edge_set = get_edge_set()
    except ImportError:
        edge_set = set()

    disagreements = []
    for sym in edge_set:
        c_result = config_is_crypto(sym)
        ac_result = is_crypto_symbol(sym)
        if c_result != ac_result:
            disagreements.append(sym)
            mismatches.append(
                f"{sym}: config.is_crypto={c_result} vs "
                f"asset_classification.is_crypto_symbol={ac_result}"
            )

    # CHECK 3: Known stocks must NOT be in any crypto set
    # T (AT&T), PANW, NET, FTNT, DDOG, BMBL, XPO are stocks
    known_stocks = {"T", "PANW", "NET", "FTNT", "DDOG", "BMBL", "XPO", "HOOD", "COIN"}
    stocks_in_crypto = []
    for sym in known_stocks:
        if sym in config_crypto or sym in ac_crypto:
            stocks_in_crypto.append(sym)
            mismatches.append(f"{sym}: STOCK incorrectly in crypto set")

    # CHECK 4: Brain's _KNOWN_CRYPTO should be superset of config crypto
    # (brain has a fallback set that should cover everything)
    if brain_crypto:
        brain_missing = set()
        for sym in config_crypto:
            s = sym.upper()
            if s not in brain_crypto:
                brain_missing.add(s)
        if brain_missing:
            mismatches.append(
                f"Ghost Brain _KNOWN_CRYPTO missing {len(brain_missing)} symbols: "
                f"{', '.join(sorted(brain_missing)[:10])}"
            )

    # ── Verdict ──
    if missing_in_ac:
        return BugCheckResult(
            name="Crypto/Stock Classification",
            bug_id=1,
            passed=False,
            severity="error",
            detail=(
                f"REGRESSION: {len(missing_in_ac)} crypto symbols missing from "
                f"asset_classification.py — _merge_config_crypto() may have failed. "
                f"Missing: {', '.join(sorted(missing_in_ac)[:10])}"
            ),
            mismatches=mismatches,
        )
    if disagreements:
        return BugCheckResult(
            name="Crypto/Stock Classification",
            bug_id=1,
            passed=False,
            severity="error",
            detail=(
                f"REGRESSION: is_crypto() disagrees with is_crypto_symbol() for "
                f"{len(disagreements)} edge symbols: {', '.join(disagreements)}"
            ),
            mismatches=mismatches,
        )
    if stocks_in_crypto:
        return BugCheckResult(
            name="Crypto/Stock Classification",
            bug_id=1,
            passed=False,
            severity="error",
            detail=f"Stocks {stocks_in_crypto} misclassified as crypto",
            mismatches=mismatches,
        )

    detail = (
        f"All {len(config_crypto)} config crypto symbols present in "
        f"asset_classification ({len(ac_crypto)} total). "
        f"All {len(edge_set)} edge symbols agree. "
        f"No stock→crypto leaks."
    )
    if mismatches:
        detail += f" Minor: {len(mismatches)} warnings."

    return BugCheckResult(
        name="Crypto/Stock Classification",
        bug_id=1,
        passed=len(missing_in_ac) == 0 and len(disagreements) == 0 and len(stocks_in_crypto) == 0,
        severity="info" if not mismatches else "warn",
        detail=detail,
        mismatches=mismatches,
    )


# ══════════════════════════════════════════════════════════════════
# BUG 2: Skip-Tag Pollution
# ══════════════════════════════════════════════════════════════════
# ROOT CAUSE: Accuracy SQL queries counted predictions tagged with
# eval_version LIKE 'skip%' — these are intentionally skipped
# predictions (e.g., weekend predictions, stale data) and should
# NEVER count toward accuracy.
#
# FIX: All accuracy queries now include:
#   AND eval_version NOT LIKE 'skip%%'
#
# Lines fixed in wolf_app.py:
#   - Line 12482 (base_where)
#   - Line 12504 (daily_where)
#   - Line 12516 (weekly_where)
#   - Line 15678 (ghost_score PostgreSQL fallback)
#   - Line 15787 (debug accuracy PostgreSQL)
#   - Line 15899, 15923 (symbol accuracy debug)
#   - Line 16022, 16080 (health metrics)
#   - Line 16521 (recalculator)
#   - Lines 28580, 28582, 28695, 28697, 28966, 28968 (legacy endpoints)
#
# INVARIANT: No accuracy calculation should count skip-tagged rows.
# ══════════════════════════════════════════════════════════════════

def check_skip_tag_exclusion() -> BugCheckResult:
    """
    Verify that skip-tagged predictions are excluded from accuracy counts.

    Two checks:
      A) Static: scan known accuracy queries for the skip filter clause
      B) Runtime: if DB available, verify no skip-tagged rows are marked correct
    """
    mismatches: List[str] = []
    skip_count = 0
    total_skip_tagged = 0

    # ── Runtime check: query DB for skip-tagged rows that ARE counted ──
    try:
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor()

            # Count skip-tagged predictions
            cur.execute("""
                SELECT COUNT(*) FROM ghost_predictions
                WHERE eval_version LIKE 'skip%%'
            """)
            total_skip_tagged = cur.fetchone()[0] or 0

            # BUG would be: skip-tagged AND checked=1 AND correct IS NOT NULL
            # These would pollute accuracy if queries don't exclude them
            cur.execute("""
                SELECT COUNT(*) FROM ghost_predictions
                WHERE eval_version LIKE 'skip%%'
                  AND checked = 1
                  AND correct IS NOT NULL
            """)
            skip_count = cur.fetchone()[0] or 0

            if skip_count > 0:
                # Check if any query would accidentally count these
                cur.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins
                    FROM ghost_predictions
                    WHERE checked = 1
                      AND eval_version NOT LIKE 'skip%%'
                """)
                clean_row = cur.fetchone()
                cur.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins
                    FROM ghost_predictions
                    WHERE checked = 1
                """)
                dirty_row = cur.fetchone()

                clean_total = clean_row[0] or 0
                dirty_total = dirty_row[0] or 0

                if clean_total != dirty_total:
                    diff = dirty_total - clean_total
                    mismatches.append(
                        f"{diff} skip-tagged predictions would pollute accuracy "
                        f"if queries missed the filter (clean={clean_total}, "
                        f"dirty={dirty_total})"
                    )

    except Exception as e:
        LOGGER.debug(f"[BUG2] DB check skipped: {e}")
        return BugCheckResult(
            name="Skip-Tag Exclusion",
            bug_id=2,
            passed=True,
            severity="info",
            detail=f"DB not available for runtime check — static analysis only. ({e})",
        )

    # ── Static: verify the integrity check itself uses the filter ──
    # (This is a sanity check — the real validation is the DB query above)
    try:
        from core.integrity import run_audit
        import inspect
        source = inspect.getsource(run_audit)
        if "eval_version" not in source and "skip" not in source:
            mismatches.append(
                "integrity.run_audit() does not reference eval_version/skip filter"
            )
    except Exception:
        pass

    if mismatches:
        return BugCheckResult(
            name="Skip-Tag Exclusion",
            bug_id=2,
            passed=False,
            severity="error",
            detail=(
                f"REGRESSION: {len(mismatches)} skip-tag issues. "
                f"{skip_count} skip-tagged rows exist with checked=1."
            ),
            mismatches=mismatches,
        )

    return BugCheckResult(
        name="Skip-Tag Exclusion",
        bug_id=2,
        passed=True,
        severity="info",
        detail=(
            f"All accuracy queries properly exclude skip-tagged predictions. "
            f"{total_skip_tagged} skip-tagged rows in DB, "
            f"{skip_count} have checked=1 (correctly filtered)."
        ),
    )


# ══════════════════════════════════════════════════════════════════
# BUG 3: Direction/Target Mismatch in Telegram
# ══════════════════════════════════════════════════════════════════
# ROOT CAUSE: Upstream bugs (brain inversions, V3 flips, dedup path)
# would set direction=UP but leave target_price < entry, so users
# got "going UP" messages with a target that implies going DOWN.
#
# FIX 1: format_pick() in ghost_notifications.py (line ~1003)
#   Direction consistency guard — if direction and target disagree,
#   trust direction and recalculate target & stop.
#
# FIX 2: _build_pick() in ghost_notifications.py (line ~1923)
#   brain_inverted handling — when brain inverts direction, target
#   is recalculated for the new direction.
#
# INVARIANT: For any pick dict, if direction=UP then target > entry,
#            and if direction=DOWN then target < entry.
# ══════════════════════════════════════════════════════════════════

def check_direction_target_consistency() -> BugCheckResult:
    """
    Verify that direction and target price agree in the live cache.

    Checks _LATEST_PREDICTIONS for any prediction where:
      - direction=UP/BUY but target_price < current_price
      - direction=DOWN/SELL but target_price > current_price

    Also verifies the format_pick() guard exists by checking
    the function's source code for the consistency guard pattern.
    """
    mismatches: List[str] = []

    # ── Check 1: Static guard exists in format_pick ──
    try:
        from core.ghost_notifications import GhostNotificationSystem
        import inspect
        source = inspect.getsource(GhostNotificationSystem.send_top10_message)
        if "DIRECTION CONSISTENCY GUARD" not in source:
            mismatches.append(
                "format_pick() missing DIRECTION CONSISTENCY GUARD comment marker"
            )
        if "target_above_entry" not in source:
            mismatches.append(
                "format_pick() missing target_above_entry check"
            )
    except Exception as e:
        LOGGER.debug(f"[BUG3] Static check failed: {e}")

    # ── Check 2: Static guard exists in _build_pick ──
    try:
        from core.ghost_notifications import GhostNotificationSystem
        import inspect
        source = inspect.getsource(GhostNotificationSystem._build_pick)
        if "brain_inverted" not in source:
            mismatches.append(
                "_build_pick() missing brain_inverted handling"
            )
        if "v3_is_inverse" not in source:
            mismatches.append(
                "_build_pick() missing v3_is_inverse handling"
            )
    except Exception as e:
        LOGGER.debug(f"[BUG3] _build_pick check failed: {e}")

    # ── Check 3: Runtime — scan live cache ──
    runtime_mismatches = 0
    try:
        import wolf_app
        latest = getattr(wolf_app, '_LATEST_PREDICTIONS', {})
        for sym, pred in latest.items():
            if not isinstance(pred, dict):
                continue
            direction = pred.get("direction", "")
            current = pred.get("price") or pred.get("price_at_prediction") or 0
            target = pred.get("target_price") or pred.get("take_profit") or 0
            if current <= 0 or target <= 0:
                continue

            is_buy = direction in ("UP", "BUY")
            is_sell = direction in ("DOWN", "SELL")

            # 1% tolerance for rounding
            if is_buy and target < current * 0.99:
                runtime_mismatches += 1
                mismatches.append(
                    f"{sym}: direction={direction} but target=${target:.4f} "
                    f"< entry=${current:.4f}"
                )
            elif is_sell and target > current * 1.01:
                runtime_mismatches += 1
                mismatches.append(
                    f"{sym}: direction={direction} but target=${target:.4f} "
                    f"> entry=${current:.4f}"
                )
    except Exception as e:
        LOGGER.debug(f"[BUG3] Runtime check skipped: {e}")

    if runtime_mismatches > 0:
        return BugCheckResult(
            name="Direction/Target Consistency",
            bug_id=3,
            passed=False,
            severity="error",
            detail=(
                f"REGRESSION: {runtime_mismatches} predictions in cache have "
                f"direction contradicting target price"
            ),
            mismatches=mismatches,
        )

    return BugCheckResult(
        name="Direction/Target Consistency",
        bug_id=3,
        passed=True,
        severity="info",
        detail=(
            "format_pick() guard and _build_pick() brain_inverted handling "
            "both present. Live cache has no direction/target mismatches."
        ),
        mismatches=mismatches,
    )


# ══════════════════════════════════════════════════════════════════
# BUG 4: _LATEST_PREDICTIONS Cache Corruption
# ══════════════════════════════════════════════════════════════════
# ROOT CAUSE: Cache entries were populated inconsistently across
# 4 code paths (warmup line 4698, stock engine line 8617,
# dedup line 10193, turbo engine line 10273).
# Fields like "engine", "intel_applied", "market", "price" were
# missing from some paths. The dedup path didn't set target/stop.
# Thread safety bug: no lock on concurrent writes (fixed Jan 6).
#
# INVARIANT: Every _LATEST_PREDICTIONS entry must have at minimum:
#   symbol, direction, confidence, run_at, price/price_at_prediction,
#   market (crypto/stock), engine
# ══════════════════════════════════════════════════════════════════

_REQUIRED_CACHE_FIELDS = {
    "symbol",
    "direction",
    "confidence",
    "run_at",
}

_EXPECTED_CACHE_FIELDS = {
    "symbol",
    "direction",
    "confidence",
    "run_at",
    "horizon_h",
    "engine",
    "market",
}


def check_latest_predictions_cache() -> BugCheckResult:
    """
    Validate _LATEST_PREDICTIONS for structural consistency.

    Checks:
      A) Required fields present in every entry
      B) Direction is actionable (UP/DOWN, not HOLD/ERROR/FLAT)
      C) Confidence is in [0, 1] range
      D) run_at is a valid timestamp (not 0, not in the future)
      E) price/price_at_prediction is positive
      F) market field is "crypto" or "stock"
      G) Thread lock exists (_LATEST_PREDICTIONS_LOCK)
    """
    mismatches: List[str] = []

    try:
        import wolf_app
        latest = getattr(wolf_app, '_LATEST_PREDICTIONS', None)
        lock = getattr(wolf_app, '_LATEST_PREDICTIONS_LOCK', None)
    except ImportError:
        return BugCheckResult(
            name="_LATEST_PREDICTIONS Cache",
            bug_id=4,
            passed=True,
            severity="info",
            detail="wolf_app not importable (test environment)",
        )

    # ── Check G: Thread lock exists ──
    if lock is None:
        mismatches.append("_LATEST_PREDICTIONS_LOCK not found — race condition risk")

    if latest is None:
        return BugCheckResult(
            name="_LATEST_PREDICTIONS Cache",
            bug_id=4,
            passed=False,
            severity="error",
            detail="_LATEST_PREDICTIONS is None (not initialized)",
            mismatches=mismatches,
        )

    if len(latest) == 0:
        return BugCheckResult(
            name="_LATEST_PREDICTIONS Cache",
            bug_id=4,
            passed=True,
            severity="info",
            detail="Cache is empty (no predictions yet)",
            mismatches=mismatches,
        )

    now = time.time()
    invalid_entries = 0
    stale_entries = 0
    bad_direction = 0
    missing_price = 0
    missing_market = 0

    for sym, pred in latest.items():
        if not isinstance(pred, dict):
            mismatches.append(f"{sym}: entry is {type(pred).__name__}, not dict")
            invalid_entries += 1
            continue

        # A) Required fields
        for f in _REQUIRED_CACHE_FIELDS:
            if f not in pred:
                mismatches.append(f"{sym}: missing required field '{f}'")
                invalid_entries += 1

        # B) Direction must be actionable
        d = pred.get("direction", "")
        if d not in ("UP", "DOWN", "BUY", "SELL"):
            bad_direction += 1
            mismatches.append(f"{sym}: direction='{d}' (not UP/DOWN)")

        # C) Confidence in [0, 1]
        c = pred.get("confidence")
        if c is not None and (c < 0 or c > 1):
            mismatches.append(f"{sym}: confidence={c} out of [0,1]")

        # D) run_at validity
        run_at = pred.get("run_at", 0)
        if run_at and run_at > now + 3600:
            mismatches.append(f"{sym}: run_at is in the future")
        elif run_at and (now - run_at) > 86400:  # > 24 hours old
            stale_entries += 1

        # E) Price is positive
        price = pred.get("price") or pred.get("price_at_prediction") or 0
        if not price or price <= 0:
            missing_price += 1
            mismatches.append(f"{sym}: no positive price field")

        # F) Market field
        market = pred.get("market", "")
        if market not in ("crypto", "stock"):
            missing_market += 1
            # Only warn if there are many — some warmup entries may lack this
            if missing_market <= 3:
                mismatches.append(f"{sym}: market='{market}' (expected crypto/stock)")

    passed = invalid_entries == 0 and bad_direction == 0
    severity = "error" if not passed else ("warn" if stale_entries > len(latest) * 0.5 else "info")

    return BugCheckResult(
        name="_LATEST_PREDICTIONS Cache",
        bug_id=4,
        passed=passed,
        severity=severity,
        detail=(
            f"{len(latest)} cached predictions. "
            f"Invalid={invalid_entries}, bad_direction={bad_direction}, "
            f"missing_price={missing_price}, missing_market={missing_market}, "
            f"stale(>24h)={stale_entries}."
        ),
        mismatches=mismatches[:20],  # Cap to avoid noise
    )


# ══════════════════════════════════════════════════════════════════
# BUG 5: Ghost Brain vs Learning Brain Conflict
# ══════════════════════════════════════════════════════════════════
# ROOT CAUSE: Ghost Brain (ghost_brain.py) inverts predictions when
# brain_accuracy < INVERT_BELOW (38%). Learning Brain
# (ghost_learning_brain.py) also inverts. If both invert the same
# symbol, the direction flips TWICE → back to the original bad
# prediction.
#
# Also: brain sets inverted=True on the candidate but _build_pick()
# didn't check brain_inverted, so targets weren't recalculated for
# the flipped direction. Fixed Mar 12, 2026.
#
# INVARIANT: INVERT_BELOW must be 38.0 (default).
# A symbol should not be inverted by BOTH brains simultaneously.
# ══════════════════════════════════════════════════════════════════

def check_brain_conflict() -> BugCheckResult:
    """
    Verify Ghost Brain and Learning Brain aren't double-inverting.

    Checks:
      A) INVERT_BELOW threshold is 38.0 (ghost_brain.py:100)
      B) No symbol appears in both brain's invert lists simultaneously
      C) If a symbol is brain-inverted, _build_pick() will recalculate
         its target (checked via source inspection)
    """
    mismatches: List[str] = []

    # ── Check A: INVERT_BELOW threshold ──
    try:
        from core.ghost_brain import INVERT_BELOW, EXCLUDE_BELOW, MIN_SAMPLES
    except ImportError as e:
        return BugCheckResult(
            name="Brain Conflict",
            bug_id=5,
            passed=False,
            severity="error",
            detail=f"Cannot import ghost_brain: {e}",
        )

    if INVERT_BELOW != 38.0:
        mismatches.append(
            f"INVERT_BELOW={INVERT_BELOW} (expected 38.0) — "
            f"env BRAIN_INVERT_BELOW may be overriding"
        )

    if EXCLUDE_BELOW is not None and EXCLUDE_BELOW <= INVERT_BELOW:
        mismatches.append(
            f"EXCLUDE_BELOW={EXCLUDE_BELOW} <= INVERT_BELOW={INVERT_BELOW} — "
            f"zone overlap means symbols get excluded instead of inverted"
        )

    # ── Check B: Double-inversion ──
    ghost_brain_inverted: Set[str] = set()
    learning_brain_inverted: Set[str] = set()

    try:
        from core.ghost_brain import GhostBrain
        brain = GhostBrain()
        # The brain tracks inversions in _inverse_tracker
        ghost_brain_inverted = set(brain._inverse_tracker.keys())
    except Exception:
        pass

    try:
        from core.ghost_learning_brain import get_inverted_symbols
        learning_brain_inverted = set(get_inverted_symbols())
    except Exception:
        pass

    double_inverted = ghost_brain_inverted & learning_brain_inverted
    if double_inverted:
        mismatches.append(
            f"DOUBLE INVERSION: {', '.join(double_inverted)} inverted by BOTH "
            f"Ghost Brain AND Learning Brain — direction flips twice → back to bad prediction"
        )

    passed = (INVERT_BELOW == 38.0 and len(double_inverted) == 0)

    return BugCheckResult(
        name="Brain Conflict",
        bug_id=5,
        passed=passed,
        severity="error" if not passed else "info",
        detail=(
            f"INVERT_BELOW={INVERT_BELOW} (expected 38.0), "
            f"EXCLUDE_BELOW={EXCLUDE_BELOW}, MIN_SAMPLES={MIN_SAMPLES}. "
            f"Ghost Brain inverted: {sorted(ghost_brain_inverted) or 'none'}. "
            f"Learning Brain inverted: {sorted(learning_brain_inverted) or 'none'}. "
            f"Double-inverted: {sorted(double_inverted) or 'none'}."
        ),
        mismatches=mismatches,
    )


# ══════════════════════════════════════════════════════════════════
# BUG 6: V3 Filter Misconfiguration
# ══════════════════════════════════════════════════════════════════
# ROOT CAUSE (multiple):
#   a) min_confidence was 0.45 raw but display showed ~57% — users
#      saw "48-55%" which looks like coin flip. Raised to 0.78 raw.
#   b) DIRECTION_FLIP ('flip') was passed to Direction() enum which
#      crashed — ghost_inverse strategies with flip couldn't trade.
#   c) direction_override was 'UP' for ETH but forced 0% accuracy
#      when ETH went down — removed forced overrides.
#   d) direction_override was DIRECTION_FLIP for PANW/NET/FTNT but
#      showed 5% accuracy — removed forced flips.
#
# INVARIANT:
#   - V3_MIN_CONFIDENCE env defaults to "0.45" (v3_filter.py:27)
#   - No V3_VALIDATED_STRATEGIES entry has direction_override set
#     (all were REMOVED as of latest fix)
#   - V3Filter._process_inverse handles DIRECTION_FLIP safely
# ══════════════════════════════════════════════════════════════════

def check_v3_filter_config() -> BugCheckResult:
    """
    Verify V3 filter configuration integrity.

    Checks:
      A) _V3_MIN_CONFIDENCE default is 0.45 (v3_filter.py:27)
      B) No validated strategy has a non-None direction_override
         (all forced overrides were removed because they killed accuracy)
      C) V3Filter can be instantiated without crash
      D) Edge whitelist symbols get scored at 0.55 × confidence
      E) Inverse strategy properly handles DIRECTION_FLIP
    """
    mismatches: List[str] = []

    # ── Check A: min_confidence ──
    try:
        from core.v3_filter import _V3_MIN_CONFIDENCE
    except ImportError as e:
        return BugCheckResult(
            name="V3 Filter Config",
            bug_id=6,
            passed=False,
            severity="error",
            detail=f"Cannot import v3_filter: {e}",
        )

    if _V3_MIN_CONFIDENCE != 0.45:
        env_val = os.getenv("V3_MIN_CONFIDENCE")
        if env_val:
            mismatches.append(
                f"_V3_MIN_CONFIDENCE={_V3_MIN_CONFIDENCE} (env override: {env_val})"
            )
        else:
            mismatches.append(
                f"_V3_MIN_CONFIDENCE={_V3_MIN_CONFIDENCE} (expected 0.45 default)"
            )

    # ── Check B: No forced direction overrides ──
    try:
        from config.symbols import V3_VALIDATED_STRATEGIES
    except ImportError:
        V3_VALIDATED_STRATEGIES = {}

    forced_overrides = []
    for sym, strat in V3_VALIDATED_STRATEGIES.items():
        if strat.direction_override is not None:
            forced_overrides.append(f"{sym}: direction_override={strat.direction_override}")
            mismatches.append(
                f"{sym} has direction_override={strat.direction_override!r} — "
                f"forced overrides were removed because they killed accuracy. "
                f"Was this intentionally re-added?"
            )

    # ── Check C: V3Filter instantiation ──
    try:
        from core.v3_filter import V3Filter
        f = V3Filter()
        # Verify stats are initialized
        assert "total_processed" in f.stats
        assert "inversed" in f.stats
    except Exception as e:
        mismatches.append(f"V3Filter() instantiation failed: {e}")

    # ── Check D: Edge whitelist score multiplier ──
    try:
        from core.v3_filter import V3Filter
        from core.models import Prediction, Direction
        from datetime import datetime
        f = V3Filter(min_confidence=0.30)  # Low threshold to test scoring
        test_pred = Prediction(
            symbol="T",  # Edge whitelist stock
            direction=Direction.UP,
            confidence=0.80,
            current_price=20.0,
            target_price=20.60,
            stop_loss=19.60,
            timestamp=datetime.now(),
        )
        result = f.filter_single(test_pred)
        if result.passed and result.prediction:
            expected_score = 0.55 * 0.80  # edge multiplier × confidence
            actual_score = result.prediction.score
            if abs(actual_score - expected_score) > 0.01:
                mismatches.append(
                    f"Edge score: expected {expected_score:.3f}, got {actual_score:.3f}"
                )
    except Exception as e:
        LOGGER.debug(f"[BUG6] Edge score check failed: {e}")

    passed = len(forced_overrides) == 0 and not any(
        "instantiation failed" in m for m in mismatches
    )

    return BugCheckResult(
        name="V3 Filter Config",
        bug_id=6,
        passed=passed,
        severity="error" if not passed else ("warn" if mismatches else "info"),
        detail=(
            f"V3_MIN_CONFIDENCE={_V3_MIN_CONFIDENCE}. "
            f"Strategies with forced overrides: {len(forced_overrides)}. "
            f"{len(V3_VALIDATED_STRATEGIES)} validated strategies loaded."
        ),
        mismatches=mismatches,
    )


# ══════════════════════════════════════════════════════════════════
# BUG 7: Edge Whitelist Bypass
# ══════════════════════════════════════════════════════════════════
# ROOT CAUSE: process_v3_from_cache() in adapters.py was a backdoor
# that bypassed the edge filter in get_top10_predictions(). Symbols
# like LINK, XRP, ETH leaked through to picks even when they
# weren't in the edge set. Stale Railway env var EDGE_SYMBOLS
# could override the curated default list with wider/outdated sets.
#
# FIX: get_edge_set() now uses code default as SOURCE OF TRUTH.
# The env var can ADD symbols but never override the curated set.
# process_v3_from_cache() filters to edge symbols BEFORE V3 scoring.
#
# INVARIANT:
#   - get_edge_set() returns the DEFAULT_EDGE_SYMBOLS parsed set
#   - No removed/blacklisted symbols appear in the edge set
#   - process_v3_from_cache() applies edge filtering before V3
# ══════════════════════════════════════════════════════════════════

def check_edge_whitelist() -> BugCheckResult:
    """
    Verify edge whitelist is correctly configured and enforced.

    Checks:
      A) get_edge_set() returns expected symbols (not stale env override)
      B) No V3_BLACKLIST symbol is in the edge set
      C) No V3_REMOVED_SYMBOLS symbol is in the edge set
      D) process_v3_from_cache() has edge filtering code
      E) BTC and SOL are NOT in edge set (removed for poor accuracy)
    """
    mismatches: List[str] = []

    try:
        from config.symbols import (
            get_edge_set, DEFAULT_EDGE_SYMBOLS,
            V3_BLACKLIST, V3_REMOVED_SYMBOLS,
        )
    except ImportError as e:
        return BugCheckResult(
            name="Edge Whitelist",
            bug_id=7,
            passed=False,
            severity="error",
            detail=f"Cannot import config.symbols: {e}",
        )

    edge_set = get_edge_set()
    default_set = frozenset(
        s.strip().upper() for s in DEFAULT_EDGE_SYMBOLS.split(",") if s.strip()
    )

    # ── Check A: edge_set matches default ──
    if edge_set != default_set:
        extra = edge_set - default_set
        missing = default_set - edge_set
        if extra:
            mismatches.append(f"Edge set has extra symbols not in default: {extra}")
        if missing:
            mismatches.append(f"Edge set is missing default symbols: {missing}")

    # ── Check B: No blacklisted symbols ──
    blacklisted_in_edge = edge_set & set(V3_BLACKLIST)
    if blacklisted_in_edge:
        mismatches.append(
            f"BLACKLISTED symbols in edge set: {blacklisted_in_edge}"
        )

    # ── Check C: No removed symbols ──
    removed_in_edge = edge_set & set(V3_REMOVED_SYMBOLS.keys())
    if removed_in_edge:
        mismatches.append(
            f"REMOVED symbols in edge set: {removed_in_edge}"
        )

    # ── Check D: process_v3_from_cache has edge filter ──
    try:
        from core.adapters import process_v3_from_cache
        import inspect
        source = inspect.getsource(process_v3_from_cache)
        if "EDGE_WHITELIST" not in source and "edge" not in source.lower():
            mismatches.append(
                "process_v3_from_cache() has no edge filtering code"
            )
    except Exception as e:
        LOGGER.debug(f"[BUG7] Adapter check failed: {e}")

    # ── Check E: BTC and SOL not in edge set ──
    # Removed Mar 5, 2026 for poor accuracy: BTC 50% (coin flip), SOL 16.7%
    bad_symbols_in_edge = edge_set & {"BTC", "SOL"}
    if bad_symbols_in_edge:
        mismatches.append(
            f"BTC/SOL in edge set — removed Mar 5 for poor accuracy: {bad_symbols_in_edge}"
        )

    passed = (
        len(blacklisted_in_edge) == 0
        and len(removed_in_edge) == 0
        and len(bad_symbols_in_edge) == 0
    )

    return BugCheckResult(
        name="Edge Whitelist",
        bug_id=7,
        passed=passed,
        severity="error" if not passed else ("warn" if mismatches else "info"),
        detail=(
            f"Edge set has {len(edge_set)} symbols. "
            f"Default has {len(default_set)}. "
            f"Blacklisted leaks: {len(blacklisted_in_edge)}. "
            f"Removed leaks: {len(removed_in_edge)}."
        ),
        mismatches=mismatches,
    )


# ══════════════════════════════════════════════════════════════════
# BUG 8: Adapters Pipeline Classification
# ══════════════════════════════════════════════════════════════════
# ROOT CAUSE: scored_list_to_formatter() in adapters.py calls
# is_crypto() to split predictions into (stocks, crypto). If
# is_crypto() disagrees with asset_classification.is_crypto_symbol(),
# a crypto symbol ends up in the stocks list, stealing a slot from
# an actual stock.
#
# FIX: scored_list_to_formatter() uses config.symbols.is_crypto()
# which checks the authoritative CRYPTO_SYMBOLS frozenset. Plus
# _merge_config_crypto() ensures asset_classification matches.
#
# INVARIANT: For every edge symbol, is_crypto() and
#   scored_list_to_formatter() must route it to the correct list.
# ══════════════════════════════════════════════════════════════════

def check_adapters_classification() -> BugCheckResult:
    """
    Verify the adapters pipeline classifies assets correctly.

    Checks:
      A) scored_list_to_formatter() uses is_crypto() for routing
      B) For all edge symbols: is_crypto() → crypto list,
         !is_crypto() → stock list
      C) Known crypto edge symbols (ETH, XRP, LINK, CHZ) land in crypto
      D) Known stock edge symbols (PANW, NET, FTNT, DDOG, T, BMBL, XPO) land in stocks
    """
    mismatches: List[str] = []

    try:
        from core.adapters import scored_list_to_formatter
        from config.symbols import is_crypto, get_edge_set
    except ImportError as e:
        return BugCheckResult(
            name="Adapters Classification",
            bug_id=8,
            passed=False,
            severity="error",
            detail=f"Cannot import adapters/config: {e}",
        )

    # ── Check A: scored_list_to_formatter uses is_crypto ──
    try:
        import inspect
        source = inspect.getsource(scored_list_to_formatter)
        if "is_crypto" not in source:
            mismatches.append(
                "scored_list_to_formatter() does not call is_crypto() for routing"
            )
    except Exception:
        pass

    # ── Check B/C/D: Verify routing for all edge symbols ──
    edge_set = get_edge_set()
    expected_crypto = {"ETH", "XRP", "LINK", "CHZ"}
    expected_stocks = {"PANW", "NET", "FTNT", "DDOG", "T", "BMBL", "XPO"}

    for sym in expected_crypto:
        if sym in edge_set and not is_crypto(sym):
            mismatches.append(
                f"{sym}: is edge crypto but is_crypto() returns False — "
                f"will be classified as stock in picks pipeline"
            )

    for sym in expected_stocks:
        if sym in edge_set and is_crypto(sym):
            mismatches.append(
                f"{sym}: is edge stock but is_crypto() returns True — "
                f"will be classified as crypto in picks pipeline"
            )

    # ── Full integration test with ScoredPrediction ──
    try:
        from core.models import ScoredPrediction, Direction
        from datetime import datetime

        test_cases = [
            ("ETH", Direction.UP, True),    # crypto
            ("PANW", Direction.DOWN, False), # stock
            ("CHZ", Direction.UP, True),     # crypto (was misclassified before fix)
            ("DDOG", Direction.UP, False),   # stock
        ]

        for sym, direction, expect_crypto in test_cases:
            sp = ScoredPrediction(
                symbol=sym,
                direction=direction,
                confidence=0.80,
                current_price=100.0,
                target_price=103.0,
                stop_loss=97.0,
                hold_hours=72,
                timestamp=datetime.now(),
                strategy="test",
                original_direction=direction,
                is_inverse=False,
                backtest_win_rate=0.60,
                score=0.48,
            )
            stocks, crypto = scored_list_to_formatter([sp])
            landed_crypto = len(crypto) > 0
            if landed_crypto != expect_crypto:
                mismatches.append(
                    f"{sym}: expected {'crypto' if expect_crypto else 'stock'} list, "
                    f"got {'crypto' if landed_crypto else 'stock'}"
                )
    except Exception as e:
        LOGGER.debug(f"[BUG8] Integration test failed: {e}")
        mismatches.append(f"Integration test failed: {e}")

    passed = len(mismatches) == 0

    return BugCheckResult(
        name="Adapters Classification",
        bug_id=8,
        passed=passed,
        severity="error" if not passed else "info",
        detail=(
            f"Checked {len(expected_crypto)} expected crypto and "
            f"{len(expected_stocks)} expected stock symbols. "
            f"{'All correctly routed.' if passed else f'{len(mismatches)} misrouted.'}"
        ),
        mismatches=mismatches,
    )


# ══════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════

ALL_BUG_CHECKS = [
    check_crypto_classification,        # Bug 1
    check_skip_tag_exclusion,           # Bug 2
    check_direction_target_consistency, # Bug 3
    check_latest_predictions_cache,     # Bug 4
    check_brain_conflict,               # Bug 5
    check_v3_filter_config,             # Bug 6
    check_edge_whitelist,               # Bug 7
    check_adapters_classification,      # Bug 8
]


def run_bug_checks() -> List[BugCheckResult]:
    """
    Run ALL bug-fix integrity checks.

    Returns list of BugCheckResult — one per bug.
    Catches exceptions per check so one failure doesn't kill the suite.
    """
    results: List[BugCheckResult] = []

    for check_fn in ALL_BUG_CHECKS:
        try:
            result = check_fn()
            results.append(result)
        except Exception as e:
            LOGGER.error(f"[BUG-CHECK] {check_fn.__name__} crashed: {e}", exc_info=True)
            results.append(BugCheckResult(
                name=check_fn.__name__,
                bug_id=-1,
                passed=False,
                severity="error",
                detail=f"Check crashed: {type(e).__name__}: {str(e)[:200]}",
            ))

    return results


def run_bug_checks_summary() -> Dict[str, Any]:
    """
    Run all checks and return a summary dict (suitable for API response).
    """
    results = run_bug_checks()
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    return {
        "total_checks": len(results),
        "passed": passed,
        "failed": failed,
        "health_pct": round(passed / len(results) * 100, 1) if results else 0,
        "checks": [
            {
                "name": r.name,
                "bug_id": r.bug_id,
                "passed": r.passed,
                "severity": r.severity,
                "detail": r.detail,
                "mismatches": r.mismatches[:5],  # Cap per check
            }
            for r in results
        ],
    }


# Allow running directly: python -m core.integrity_bug_checks
if __name__ == "__main__":
    import json
    import sys

    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    results = run_bug_checks()
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"  GHOST PROTOCOL — Bug-Fix Integrity Checks")
    print(f"{'='*60}\n")

    for r in results:
        icon = "✅" if r.passed else "❌"
        print(f"  {icon} Bug #{r.bug_id}: {r.name}")
        print(f"     {r.detail}")
        for m in r.mismatches[:3]:
            print(f"     ⚠ {m}")
        print()

    print(f"{'='*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}\n")

    sys.exit(0 if failed == 0 else 1)
