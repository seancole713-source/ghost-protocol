#!/usr/bin/env python3
"""
🧠 Brain Data Layer — Rich Context for Ghost Brain v3
======================================================

The brain itself has NO database dependency. This module:
  1. Defines data structures (BrainContext, SymbolContext)
  2. Queries PostgreSQL for all data the brain needs
  3. Packages it into a BrainContext for the brain to consume

Data sources queried:
  ┌─────────────────────────────────┬──────────────────────────────────┐
  │ ghost_symbol_accuracy           │ Base per-symbol accuracy         │
  │ ghost_prediction_outcomes       │ Direction split, recency,        │
  │                                 │ calibration, magnitude, DOW      │
  │ ghost_symbol_trust              │ Streaks, trust levels            │
  └─────────────────────────────────┴──────────────────────────────────┘

Usage (production):
    from core.brain_data import load_brain_context
    context = await load_brain_context(db_url, symbols, market_data)
    decisions = brain.analyze_batch(predictions, context=context)

Usage (testing):
    context = BrainContext(symbols={"BTC": SymbolContext(...)})
    decisions = brain.analyze_batch(predictions, context=context)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

LOGGER = logging.getLogger("brain_data")


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SymbolContext:
    """Rich per-symbol data for brain decisions.

    Populated by load_brain_context() in production,
    or directly in tests with mock values.
    """

    # ── Base accuracy (from ghost_symbol_accuracy) ──
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy_pct: float = 50.0

    # ── #1: Per-direction accuracy ──
    up_total: int = 0
    up_correct: int = 0
    up_accuracy: float = 50.0
    down_total: int = 0
    down_correct: int = 0
    down_accuracy: float = 50.0

    # ── #2: Recent accuracy (last N days) ──
    recent_total: int = 0
    recent_correct: int = 0
    recent_accuracy: float = 50.0

    # ── #4: Streak data (from ghost_symbol_trust) ──
    trust_level: int = 1             # 1=unproven, 2=warming, 3=proven
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    current_streak: int = 0          # positive=wins, negative=losses

    # ── #6: Move magnitude ──
    avg_win_magnitude: float = 0.0   # avg |price_change_pct| on correct
    avg_loss_magnitude: float = 0.0  # avg |price_change_pct| on wrong

    # ── #7: Day-of-week accuracy ──
    # {0: 62.5, 1: 58.3, ...} where 0=Sunday, 1=Monday, ... 6=Saturday
    dow_accuracy: Dict[int, float] = field(default_factory=dict)
    dow_totals: Dict[int, int] = field(default_factory=dict)

    # ── #14: Prune tracking ──
    days_tracked: int = 0            # days since first prediction
    first_prediction_date: Optional[datetime] = None

    # ── #19: Expected value ──
    expected_value: float = 0.0      # avg_win * win_rate - avg_loss * loss_rate


@dataclass
class BrainContext:
    """Complete context package for one brain analysis cycle.

    Contains all data the brain needs to make optimal decisions
    across all 25 cognitive abilities. Populated once per cycle
    by load_brain_context(), then passed to brain.analyze_batch().
    """

    # ── Per-symbol enriched data ──
    symbols: Dict[str, SymbolContext] = field(default_factory=dict)

    # ── #3: Confidence calibration curve ──
    # Maps confidence bucket string to actual hit rate
    # e.g. {"0.5": 0.48, "0.6": 0.55, "0.7": 0.61, "0.8": 0.63}
    calibration_curve: Dict[str, float] = field(default_factory=dict)

    # ── #5: Market regime ──
    market_regime: str = "unknown"    # calm/neutral/elevated/fear/panic
    vix_level: float = 0.0

    # ── #10: Fear & Greed ──
    fear_greed_index: int = 50        # 0-100

    # ── #18: Cross-asset context ──
    btc_24h_change: float = 0.0      # BTC % change last 24h
    eth_24h_change: float = 0.0      # ETH % change last 24h
    spy_24h_change: float = 0.0      # SPY % change last 24h

    # ── #7, #20: Time context ──
    current_day: int = 0              # 0=Sunday .. 6=Saturday (SQL DOW)
    is_weekend: bool = False
    current_hour: int = 12

    # ── #23: Circuit breaker data ──
    rolling_3d_accuracy: float = 50.0
    rolling_3d_total: int = 0

    # ── Global stats ──
    total_symbols_tracked: int = 0
    avg_global_accuracy: float = 50.0


# ═══════════════════════════════════════════════════════════════════
# SQL QUERIES
# ═══════════════════════════════════════════════════════════════════

# #1: Per-direction accuracy
SQL_DIRECTION_ACCURACY = """
SELECT
    symbol,
    predicted_direction,
    COUNT(*) as total,
    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct
FROM ghost_prediction_outcomes
WHERE hit_direction IS NOT NULL
GROUP BY symbol, predicted_direction
"""

# #2: Recent accuracy (last N days)
SQL_RECENT_ACCURACY = """
SELECT
    symbol,
    COUNT(*) as total,
    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct
FROM ghost_prediction_outcomes
WHERE hit_direction IS NOT NULL
  AND created_at > NOW() - INTERVAL '%s days'
GROUP BY symbol
"""

# #3: Confidence calibration curve
SQL_CALIBRATION = """
SELECT
    FLOOR(predicted_confidence * 10) / 10.0 as conf_bucket,
    COUNT(*) as total,
    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct
FROM ghost_prediction_outcomes
WHERE hit_direction IS NOT NULL
  AND predicted_confidence IS NOT NULL
GROUP BY conf_bucket
HAVING COUNT(*) >= 10
ORDER BY conf_bucket
"""

# #6: Move magnitude per symbol
SQL_MAGNITUDE = """
SELECT
    symbol,
    AVG(CASE WHEN hit_direction = 1 THEN ABS(price_change_pct) ELSE NULL END) as avg_win_mag,
    AVG(CASE WHEN hit_direction = 0 THEN ABS(price_change_pct) ELSE NULL END) as avg_loss_mag
FROM ghost_prediction_outcomes
WHERE hit_direction IS NOT NULL
  AND price_change_pct IS NOT NULL
GROUP BY symbol
"""

# #7: Day-of-week accuracy
SQL_DOW = """
SELECT
    symbol,
    EXTRACT(DOW FROM created_at)::int as dow,
    COUNT(*) as total,
    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct
FROM ghost_prediction_outcomes
WHERE hit_direction IS NOT NULL
GROUP BY symbol, dow
"""

# #4: Streak data from trust table
SQL_TRUST = """
SELECT
    symbol, trust_level, consecutive_wins, consecutive_losses,
    total_predictions, total_wins
FROM ghost_symbol_trust
"""

# #14: First prediction date per symbol
SQL_FIRST_PREDICTION = """
SELECT
    symbol,
    MIN(created_at) as first_date
FROM ghost_prediction_outcomes
GROUP BY symbol
"""

# #23: Rolling 3-day accuracy
SQL_ROLLING_3D = """
SELECT
    COUNT(*) as total,
    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct
FROM ghost_prediction_outcomes
WHERE hit_direction IS NOT NULL
  AND created_at > NOW() - INTERVAL '3 days'
"""

# Base accuracy (from ghost_symbol_accuracy)
SQL_BASE_ACCURACY = """
SELECT symbol, total_predictions, correct_predictions, accuracy_pct
FROM ghost_symbol_accuracy
"""


# ═══════════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════════

async def load_brain_context(
    db_url: str,
    symbols: Optional[List[str]] = None,
    market_data: Optional[Dict[str, Any]] = None,
    recency_days: int = 30,
) -> BrainContext:
    """
    Load complete brain context from PostgreSQL.

    Queries 7 tables/views in parallel to build a rich SymbolContext
    for every tracked symbol. This runs once per notification cycle
    (~every 2 hours) so query overhead is negligible.

    Args:
        db_url:        PostgreSQL connection string
        symbols:       Optional filter (None = all symbols)
        market_data:   Dict with live market context:
                       {"regime": "calm", "vix": 15.2, "fear_greed": 55,
                        "btc_24h": -2.3, "spy_24h": 0.4}
        recency_days:  Window for recent accuracy calculation

    Returns:
        BrainContext with all data populated
    """
    import asyncpg
    from datetime import datetime

    ctx = BrainContext()
    market_data = market_data or {}

    # Fill market context
    ctx.market_regime = market_data.get("regime", "unknown")
    ctx.vix_level = market_data.get("vix", 0.0)
    ctx.fear_greed_index = market_data.get("fear_greed", 50)
    ctx.btc_24h_change = market_data.get("btc_24h", 0.0)
    ctx.eth_24h_change = market_data.get("eth_24h", 0.0)
    ctx.spy_24h_change = market_data.get("spy_24h", 0.0)

    now = datetime.now()
    ctx.current_day = now.weekday()  # 0=Monday in Python
    # Convert to SQL DOW (0=Sunday) for DOW matching
    ctx.current_day = (now.weekday() + 1) % 7  # 0=Sun,1=Mon..6=Sat
    ctx.is_weekend = now.weekday() >= 5  # Sat=5, Sun=6
    ctx.current_hour = now.hour

    try:
        conn = await asyncpg.connect(db_url)
    except Exception as exc:
        LOGGER.error(f"[BRAIN_DATA] DB connection failed: {exc}")
        return ctx

    try:
        # ── 1. Base accuracy ──
        try:
            rows = await conn.fetch(SQL_BASE_ACCURACY)
            for row in rows:
                sym = row["symbol"].upper()
                sc = ctx.symbols.setdefault(sym, SymbolContext())
                sc.total_predictions = row["total_predictions"] or 0
                sc.correct_predictions = row["correct_predictions"] or 0
                sc.accuracy_pct = float(row["accuracy_pct"] or 50.0)
        except Exception as exc:
            LOGGER.warning(f"[BRAIN_DATA] Base accuracy query failed: {exc}")

        # ── 2. Direction split (#1) ──
        try:
            rows = await conn.fetch(SQL_DIRECTION_ACCURACY)
            for row in rows:
                sym = row["symbol"].upper()
                sc = ctx.symbols.setdefault(sym, SymbolContext())
                direction = (row["predicted_direction"] or "").upper()
                total = row["total"] or 0
                correct = row["correct"] or 0
                acc = (correct / total * 100.0) if total > 0 else 50.0
                if direction == "UP":
                    sc.up_total = total
                    sc.up_correct = correct
                    sc.up_accuracy = acc
                elif direction == "DOWN":
                    sc.down_total = total
                    sc.down_correct = correct
                    sc.down_accuracy = acc
        except Exception as exc:
            LOGGER.warning(f"[BRAIN_DATA] Direction accuracy query failed: {exc}")

        # ── 3. Recent accuracy (#2) ──
        try:
            query = SQL_RECENT_ACCURACY % recency_days
            rows = await conn.fetch(query)
            for row in rows:
                sym = row["symbol"].upper()
                sc = ctx.symbols.setdefault(sym, SymbolContext())
                sc.recent_total = row["total"] or 0
                sc.recent_correct = row["correct"] or 0
                sc.recent_accuracy = (
                    (sc.recent_correct / sc.recent_total * 100.0)
                    if sc.recent_total > 0 else sc.accuracy_pct
                )
        except Exception as exc:
            LOGGER.warning(f"[BRAIN_DATA] Recent accuracy query failed: {exc}")

        # ── 4. Confidence calibration (#3) ──
        try:
            rows = await conn.fetch(SQL_CALIBRATION)
            for row in rows:
                bucket = row["conf_bucket"]
                total = row["total"] or 0
                correct = row["correct"] or 0
                if total >= 10:
                    key = f"{float(bucket):.1f}"
                    ctx.calibration_curve[key] = correct / total
        except Exception as exc:
            LOGGER.warning(f"[BRAIN_DATA] Calibration query failed: {exc}")

        # ── 5. Magnitude (#6) ──
        try:
            rows = await conn.fetch(SQL_MAGNITUDE)
            for row in rows:
                sym = row["symbol"].upper()
                sc = ctx.symbols.setdefault(sym, SymbolContext())
                sc.avg_win_magnitude = float(row["avg_win_mag"] or 0.0)
                sc.avg_loss_magnitude = float(row["avg_loss_mag"] or 0.0)
                # Compute expected value (#19)
                if sc.total_predictions > 0:
                    win_rate = sc.accuracy_pct / 100.0
                    sc.expected_value = (
                        sc.avg_win_magnitude * win_rate
                        - sc.avg_loss_magnitude * (1.0 - win_rate)
                    )
        except Exception as exc:
            LOGGER.warning(f"[BRAIN_DATA] Magnitude query failed: {exc}")

        # ── 6. Day-of-week (#7) ──
        try:
            rows = await conn.fetch(SQL_DOW)
            for row in rows:
                sym = row["symbol"].upper()
                sc = ctx.symbols.setdefault(sym, SymbolContext())
                dow = row["dow"]  # 0=Sunday in SQL
                total = row["total"] or 0
                correct = row["correct"] or 0
                if total >= 5:
                    sc.dow_accuracy[dow] = correct / total * 100.0
                    sc.dow_totals[dow] = total
        except Exception as exc:
            LOGGER.warning(f"[BRAIN_DATA] DOW query failed: {exc}")

        # ── 7. Trust/streak data (#4) ──
        try:
            rows = await conn.fetch(SQL_TRUST)
            for row in rows:
                sym = row["symbol"].upper()
                sc = ctx.symbols.setdefault(sym, SymbolContext())
                sc.trust_level = row["trust_level"] or 1
                sc.consecutive_wins = row["consecutive_wins"] or 0
                sc.consecutive_losses = row["consecutive_losses"] or 0
                sc.current_streak = (
                    sc.consecutive_wins if sc.consecutive_wins > 0
                    else -sc.consecutive_losses
                )
        except Exception as exc:
            LOGGER.warning(f"[BRAIN_DATA] Trust query failed: {exc}")

        # ── 8. First prediction date (#14) ──
        try:
            rows = await conn.fetch(SQL_FIRST_PREDICTION)
            for row in rows:
                sym = row["symbol"].upper()
                sc = ctx.symbols.setdefault(sym, SymbolContext())
                first_date = row["first_date"]
                if first_date:
                    sc.first_prediction_date = first_date
                    sc.days_tracked = (now - first_date).days
        except Exception as exc:
            LOGGER.warning(f"[BRAIN_DATA] First prediction query failed: {exc}")

        # ── 9. Rolling 3-day accuracy (#23) ──
        try:
            row = await conn.fetchrow(SQL_ROLLING_3D)
            if row:
                total = row["total"] or 0
                correct = row["correct"] or 0
                ctx.rolling_3d_total = total
                ctx.rolling_3d_accuracy = (
                    correct / total * 100.0 if total > 0 else 50.0
                )
        except Exception as exc:
            LOGGER.warning(f"[BRAIN_DATA] Rolling 3d query failed: {exc}")

        # ── Global stats ──
        ctx.total_symbols_tracked = len(ctx.symbols)
        if ctx.symbols:
            accs = [
                s.accuracy_pct for s in ctx.symbols.values()
                if s.total_predictions > 0
            ]
            ctx.avg_global_accuracy = sum(accs) / len(accs) if accs else 50.0

    except Exception as exc:
        LOGGER.error(f"[BRAIN_DATA] Unexpected error: {exc}")
    finally:
        await conn.close()

    LOGGER.info(
        f"[BRAIN_DATA] Loaded context: {len(ctx.symbols)} symbols, "
        f"{len(ctx.calibration_curve)} calibration buckets, "
        f"regime={ctx.market_regime}, F&G={ctx.fear_greed_index}"
    )
    return ctx


def build_context_from_accuracy_data(
    accuracy_data: Dict[str, Dict],
) -> BrainContext:
    """
    Build a minimal BrainContext from the old-style accuracy_data dict.

    This provides backward compatibility: if the full DB loader can't
    run (tests, offline mode), the brain still works with basic data.

    Args:
        accuracy_data: Dict from get_symbol_accuracy_from_postgres()
                       {symbol: {total, correct, accuracy_pct}}

    Returns:
        BrainContext with base accuracy only (no enrichment)
    """
    ctx = BrainContext()
    for symbol, data in accuracy_data.items():
        sc = SymbolContext(
            total_predictions=data.get("total", 0),
            correct_predictions=data.get("correct", 0),
            accuracy_pct=data.get("accuracy_pct", 50.0),
        )
        # Fill direction/recent with base accuracy (no enrichment)
        sc.up_accuracy = sc.accuracy_pct
        sc.down_accuracy = sc.accuracy_pct
        sc.recent_accuracy = sc.accuracy_pct
        ctx.symbols[symbol.upper()] = sc

    ctx.total_symbols_tracked = len(ctx.symbols)
    if ctx.symbols:
        accs = [
            s.accuracy_pct for s in ctx.symbols.values()
            if s.total_predictions > 0
        ]
        ctx.avg_global_accuracy = sum(accs) / len(accs) if accs else 50.0

    return ctx
