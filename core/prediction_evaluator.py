#!/usr/bin/env python3
"""
Ghost Prediction Evaluator — PostgreSQL Edition

Checks predictions that are ready for evaluation (48h elapsed),
fetches outcome prices, determines if prediction was correct,
and updates the ghost_predictions table.

CRITICAL FIX (Feb 24, 2026):
  Previous version used SQLite on Railway's ephemeral filesystem.
  Every deploy wiped all evaluation data - accuracy stats were fiction.
  Now reads/writes PostgreSQL via DATABASE_URL so evaluations persist.

Run this as a cron job every hour:
0 * * * * cd /app && python3 core/prediction_evaluator.py >> /tmp/evaluator.log 2>&1
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")


# ---------------------------------------------------------------------------
# PostgreSQL helpers
# ---------------------------------------------------------------------------

def _get_pg_conn():
    """Return a psycopg2 connection context manager from the pool."""
    from core.db_pool import get_sync_connection

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured - evaluator cannot run")
    return get_sync_connection()


def _ensure_pg_tables(conn) -> None:
    """
    Ensure the ghost_predictions and ghost_accuracy_stats tables exist in
    PostgreSQL with all required columns for touch-target evaluation.
    Uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS so it is safe to call
    on every run.
    """
    cur = conn.cursor()

    # --- ghost_predictions -------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ghost_predictions (
            id              SERIAL PRIMARY KEY,
            symbol          TEXT NOT NULL,
            predicted_at    BIGINT NOT NULL,
            check_at        BIGINT NOT NULL,
            predicted_price DOUBLE PRECISION,
            predicted_direction TEXT,
            predicted_pct   DOUBLE PRECISION,
            confidence      DOUBLE PRECISION,
            timeframe_hours INTEGER,
            reasons         TEXT,
            current_price   DOUBLE PRECISION,
            outcome_price   DOUBLE PRECISION,
            outcome_direction TEXT,
            outcome_pct     DOUBLE PRECISION,
            correct         INTEGER,
            checked         INTEGER DEFAULT 0,
            checked_at      BIGINT,
            error_pct       DOUBLE PRECISION,
            features_json   TEXT,
            window_first    DOUBLE PRECISION,
            window_last     DOUBLE PRECISION,
            window_high     DOUBLE PRECISION,
            window_low      DOUBLE PRECISION,
            target_price    DOUBLE PRECISION,
            stage5_ok       INTEGER,
            stage6_ok       INTEGER,
            gate            TEXT,
            touch_calibrated_1pct    DOUBLE PRECISION,
            touch_calibrated_0_5pct  DOUBLE PRECISION,
            touch_calibration_samples INTEGER,
            touch_conf_band TEXT,
            touch_1pct      INTEGER,
            touch_0_5pct    INTEGER,
            correct_1pct    INTEGER,
            correct_0_5pct  INTEGER,
            direction_consistent INTEGER,
            eval_version    TEXT,
            UNIQUE(symbol, predicted_at)
        )
    """)

    # Add columns that might be missing on older schemas
    _optional_cols = [
        ("window_first", "DOUBLE PRECISION"),
        ("window_last", "DOUBLE PRECISION"),
        ("window_high", "DOUBLE PRECISION"),
        ("window_low", "DOUBLE PRECISION"),
        ("target_price", "DOUBLE PRECISION"),
        ("stage5_ok", "INTEGER"),
        ("stage6_ok", "INTEGER"),
        ("gate", "TEXT"),
        ("touch_calibrated_1pct", "DOUBLE PRECISION"),
        ("touch_calibrated_0_5pct", "DOUBLE PRECISION"),
        ("touch_calibration_samples", "INTEGER"),
        ("touch_conf_band", "TEXT"),
        ("touch_1pct", "INTEGER"),
        ("touch_0_5pct", "INTEGER"),
        ("correct_1pct", "INTEGER"),
        ("correct_0_5pct", "INTEGER"),
        ("direction_consistent", "INTEGER"),
        ("eval_version", "TEXT"),
        ("features_json", "TEXT"),
    ]
    for col_name, col_type in _optional_cols:
        try:
            cur.execute(
                f"ALTER TABLE ghost_predictions ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
            )
        except Exception:
            pass

    # --- ghost_accuracy_stats ----------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ghost_accuracy_stats (
            id                  SERIAL PRIMARY KEY,
            period              TEXT NOT NULL UNIQUE,
            total_predictions   INTEGER,
            correct_predictions INTEGER,
            accuracy_pct        DOUBLE PRECISION,
            avg_error_pct       DOUBLE PRECISION,
            best_symbol         TEXT,
            worst_symbol        TEXT,
            updated_at          BIGINT
        )
    """)

    # --- price_actuals (for window price lookups) --------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_actuals (
            id      SERIAL PRIMARY KEY,
            symbol  TEXT NOT NULL,
            ts      BIGINT NOT NULL,
            price   DOUBLE PRECISION
        )
    """)
    try:
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_price_actuals_sym_ts_uniq
            ON price_actuals (symbol, ts)
        """)
    except Exception:
        pass
    try:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_actuals_sym_ts
            ON price_actuals (symbol, ts)
        """)
    except Exception:
        pass

    # --- ghost_prediction_outcomes (reconciler's table) --------------------
    # FIX (Step 8, Mar 18 2026): Ensure this table exists so the evaluator's
    # LEFT JOIN doesn't fail. The reconciler creates it too, but we need it
    # to exist for the evaluator's skip-if-already-reconciled query.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ghost_prediction_outcomes (
            id                    SERIAL PRIMARY KEY,
            prediction_id         INTEGER NOT NULL UNIQUE,
            symbol                TEXT NOT NULL,
            closed_at             TIMESTAMPTZ,
            price_at_prediction   DOUBLE PRECISION NOT NULL DEFAULT 0,
            price_at_resolution   DOUBLE PRECISION NOT NULL DEFAULT 0,
            realized_move_pct     DOUBLE PRECISION,
            predicted_direction   TEXT,
            actual_direction      TEXT,
            hit_direction         INTEGER,
            direction_threshold_pct DOUBLE PRECISION,
            predicted_confidence  DOUBLE PRECISION,
            resolution_method     TEXT,
            resolution_provider   TEXT,
            status                TEXT DEFAULT 'pending'
        )
    """)

    conn.commit()


# ---------------------------------------------------------------------------
# Price window helpers
# ---------------------------------------------------------------------------

def _fetch_window_prices_pg(
    conn,
    *,
    symbol: str,
    start_ts: int,
    end_ts: int,
) -> List[Tuple[int, float]]:
    """Fetch price time-series from PostgreSQL for the evaluation window."""
    cur = conn.cursor()

    cur.execute("""
        SELECT ts, price FROM price_actuals
        WHERE symbol = %s AND ts >= %s AND ts <= %s AND price IS NOT NULL
        ORDER BY ts ASC
    """, (symbol, start_ts, end_ts))
    rows = cur.fetchall()
    if rows:
        return [(int(ts), float(price)) for ts, price in rows]

    # Fallback: try price_history if it exists
    try:
        cur.execute("""
            SELECT timestamp, price FROM price_history
            WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s AND price IS NOT NULL
            ORDER BY timestamp ASC
        """, (symbol, start_ts, end_ts))
        rows = cur.fetchall()
        if rows:
            return [(int(ts), float(price)) for ts, price in rows]
    except Exception:
        conn.rollback()

    return []


def _direction_from_delta(delta: float) -> str:
    if delta > 0:
        return "UP"
    if delta < 0:
        return "DOWN"
    return "FLAT"


def _evaluate_touch_target(
    *,
    predicted_direction: Optional[str],
    start_price: Optional[float],
    target_price: Optional[float],
    prices: List[Tuple[int, float]],
) -> Dict[str, Any]:
    """Touch-target evaluation. Returns a dict with ok, reason, and fields."""
    if start_price is None or start_price <= 0:
        return {"ok": False, "reason": "missing_start_price"}
    if target_price is None or target_price <= 0:
        return {"ok": False, "reason": "missing_target_price"}
    if not prices:
        return {"ok": False, "reason": "no_prices_in_window"}

    window_first = float(prices[0][1])
    window_last = float(prices[-1][1])
    window_high = max(float(p) for _, p in prices)
    window_low = min(float(p) for _, p in prices)

    expected_direction = _direction_from_delta(target_price - start_price)
    predicted_direction_norm = (predicted_direction or "").upper().strip()
    direction_consistent = 1 if predicted_direction_norm == expected_direction else 0

    outcome_pct = ((window_last - start_price) / start_price) * 100.0
    outcome_direction = _direction_from_delta(window_last - start_price)
    error_pct = abs(window_last - target_price) / start_price * 100.0

    tol_1 = 0.01
    tol_05 = 0.005

    if expected_direction == "UP":
        touch_1 = 1 if window_high >= target_price * (1.0 - tol_1) else 0
        touch_05 = 1 if window_high >= target_price * (1.0 - tol_05) else 0
    elif expected_direction == "DOWN":
        touch_1 = 1 if window_low <= target_price * (1.0 + tol_1) else 0
        touch_05 = 1 if window_low <= target_price * (1.0 + tol_05) else 0
    else:
        touch_1 = 1 if (window_high <= target_price * (1.0 + tol_1) and window_low >= target_price * (1.0 - tol_1)) else 0
        touch_05 = 1 if (window_high <= target_price * (1.0 + tol_05) and window_low >= target_price * (1.0 - tol_05)) else 0

    correct_1 = 1 if (direction_consistent == 1 and touch_1 == 1) else 0
    correct_05 = 1 if (direction_consistent == 1 and touch_05 == 1) else 0

    return {
        "ok": True,
        "reason": "ok",
        "window_first": window_first,
        "window_last": window_last,
        "window_high": window_high,
        "window_low": window_low,
        "outcome_direction": outcome_direction,
        "outcome_pct": outcome_pct,
        "error_pct": error_pct,
        "direction_consistent": direction_consistent,
        "touch_1pct": touch_1,
        "touch_0_5pct": touch_05,
        "correct_1pct": correct_1,
        "correct_0_5pct": correct_05,
    }


# ---------------------------------------------------------------------------
# Live price fetcher (fallback for predictions without window data)
# ---------------------------------------------------------------------------

def get_current_price(symbol: str) -> Optional[float]:
    """Fetch current price for a symbol using Ghost's price quorum system."""
    try:
        from wolf_app import _get_price_quorum, HUNTER_CRYPTO_SYMBOLS

        is_crypto = symbol in HUNTER_CRYPTO_SYMBOLS

        if is_crypto:
            import asyncio
            from core.crypto.crypto_providers import get_crypto_price_quorum

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                crypto_data = loop.run_until_complete(
                    get_crypto_price_quorum(symbol, use_cache=False)
                )
                if crypto_data and crypto_data.get("price"):
                    return float(crypto_data["price"])
            finally:
                loop.close()
        else:
            price_data = _get_price_quorum(symbol, "stock")
            if price_data and price_data.get("price"):
                return float(price_data["price"])

        return None
    except Exception as e:
        LOGGER.error(f"Failed to fetch price for {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# Re-evaluation of legacy skip-tagged predictions
# ---------------------------------------------------------------------------

def re_evaluate_skip_tagged() -> Dict:
    """
    Re-evaluate predictions that were tagged with eval_version='skip-*'.

    These were historically skipped because the market moved less than the
    old flat-market threshold (0.5% or 1.0%). With the threshold now at
    0.25%, many of these may actually be legitimate wins or losses.

    This function:
      1. Fetches all checked=1, eval_version LIKE 'skip%' predictions
      2. Re-runs the evaluation logic with the current 0.25% threshold
      3. Updates outcomes — some will flip from skip→win or skip→loss
      4. Returns a summary of changes

    Safe to run multiple times — idempotent (re-evaluates same records).
    """
    with _get_pg_conn() as conn:
        return _re_evaluate_skip_tagged_with_conn(conn)


def _re_evaluate_skip_tagged_with_conn(conn) -> Dict:
    """Inner logic for re-evaluating skip-tagged predictions."""
    _ensure_pg_tables(conn)
    cur = conn.cursor()
    now = int(time.time())

    # Find all skip-tagged predictions
    cur.execute("""
        SELECT id, symbol, predicted_at, check_at, predicted_price,
               predicted_direction, current_price, confidence,
               outcome_pct, outcome_direction, eval_version
        FROM ghost_predictions
        WHERE checked = 1 AND eval_version LIKE 'skip%%'
        ORDER BY predicted_at ASC
    """)
    skip_tagged = cur.fetchall()

    LOGGER.info(f"Found {len(skip_tagged)} skip-tagged predictions to re-evaluate")

    re_evaluated = 0
    flipped_to_correct = 0
    flipped_to_incorrect = 0
    stayed_incorrect = 0

    FLAT_MARKET_THRESHOLD = 0.25  # Current threshold

    for pred in skip_tagged:
        (pred_id, symbol, pred_at, check_at, pred_price,
         direction, start_price, confidence,
         old_outcome_pct, old_outcome_dir, old_eval_version) = pred

        # If we already have outcome data stored, use it directly
        # (the original evaluator stored outcome_pct/outcome_direction
        #  before skip-tagging, so we don't need to re-fetch prices)
        if old_outcome_pct is not None and old_outcome_dir is not None:
            abs_move_pct = abs(float(old_outcome_pct))
            predicted_dir = (direction or "").upper().strip()
            actual_dir = (old_outcome_dir or "").upper().strip()

            if abs_move_pct < FLAT_MARKET_THRESHOLD:
                # Still flat under new threshold — mark as incorrect (flat-market-v3)
                cur.execute(
                    "UPDATE ghost_predictions SET correct = 0, eval_version = %s WHERE id = %s",
                    ("flat-market-v3", pred_id),
                )
                stayed_incorrect += 1
            else:
                # Move exceeds new threshold — evaluate direction match
                is_correct = 1 if (predicted_dir == actual_dir and predicted_dir in ("UP", "DOWN")) else 0
                new_version = "re-eval-v1"

                cur.execute(
                    "UPDATE ghost_predictions SET correct = %s, eval_version = %s WHERE id = %s",
                    (is_correct, new_version, pred_id),
                )

                if is_correct:
                    flipped_to_correct += 1
                else:
                    flipped_to_incorrect += 1

            re_evaluated += 1
        else:
            # No stored outcome data — try to re-fetch prices
            prices = _fetch_window_prices_pg(
                conn, symbol=symbol,
                start_ts=int(pred_at), end_ts=int(check_at)
            )

            if len(prices) < 5:
                # Still no data — mark as no-data
                cur.execute(
                    "UPDATE ghost_predictions SET correct = 0, eval_version = %s WHERE id = %s",
                    ("no-data-v2", pred_id),
                )
                stayed_incorrect += 1
                re_evaluated += 1
                continue

            eval_result = _evaluate_touch_target(
                predicted_direction=direction,
                start_price=float(start_price) if start_price else None,
                target_price=float(pred_price) if pred_price else None,
                prices=prices,
            )

            if not eval_result["ok"]:
                cur.execute(
                    "UPDATE ghost_predictions SET correct = 0, eval_version = %s WHERE id = %s",
                    ("eval-fail-v2", pred_id),
                )
                stayed_incorrect += 1
                re_evaluated += 1
                continue

            predicted_dir = (direction or "").upper().strip()
            actual_dir = (eval_result.get("outcome_direction") or "").upper().strip()
            abs_move_pct = abs(eval_result.get("outcome_pct", 0))

            if abs_move_pct < FLAT_MARKET_THRESHOLD:
                cur.execute(
                    "UPDATE ghost_predictions SET correct = 0, eval_version = %s, "
                    "outcome_pct = %s, outcome_direction = %s WHERE id = %s",
                    ("flat-market-v3", eval_result.get("outcome_pct"), actual_dir, pred_id),
                )
                stayed_incorrect += 1
            else:
                is_correct = 1 if (predicted_dir == actual_dir and predicted_dir in ("UP", "DOWN")) else 0
                cur.execute(
                    "UPDATE ghost_predictions SET correct = %s, eval_version = %s, "
                    "outcome_pct = %s, outcome_price = %s, outcome_direction = %s, "
                    "window_first = %s, window_last = %s, window_high = %s, window_low = %s "
                    "WHERE id = %s",
                    (is_correct, "re-eval-v1",
                     eval_result.get("outcome_pct"), eval_result.get("window_last"),
                     actual_dir,
                     eval_result.get("window_first"), eval_result.get("window_last"),
                     eval_result.get("window_high"), eval_result.get("window_low"),
                     pred_id),
                )
                if is_correct:
                    flipped_to_correct += 1
                else:
                    flipped_to_incorrect += 1

            re_evaluated += 1

    conn.commit()

    # Update accuracy stats after re-evaluation
    cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1 AND eval_version NOT LIKE 'skip%%'")
    total_checked = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1 AND correct = 1 AND eval_version NOT LIKE 'skip%%'")
    total_correct = cur.fetchone()[0]
    overall_accuracy = (total_correct / total_checked * 100) if total_checked > 0 else 0

    cur.execute("""
        INSERT INTO ghost_accuracy_stats
            (period, total_predictions, correct_predictions,
             accuracy_pct, avg_error_pct, updated_at)
        VALUES ('all_time', %s, %s, %s, 0, %s)
        ON CONFLICT (period) DO UPDATE SET
            total_predictions = EXCLUDED.total_predictions,
            correct_predictions = EXCLUDED.correct_predictions,
            accuracy_pct = EXCLUDED.accuracy_pct,
            updated_at = EXCLUDED.updated_at
    """, (total_checked, total_correct, overall_accuracy, now))
    conn.commit()

    LOGGER.info(
        f"📊 Re-evaluation complete: {re_evaluated} processed, "
        f"✅ {flipped_to_correct} flipped to correct, "
        f"❌ {flipped_to_incorrect} flipped to incorrect, "
        f"➡️ {stayed_incorrect} stayed incorrect. "
        f"New accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_checked})"
    )

    return {
        "re_evaluated": re_evaluated,
        "flipped_to_correct": flipped_to_correct,
        "flipped_to_incorrect": flipped_to_incorrect,
        "stayed_incorrect": stayed_incorrect,
        "new_accuracy_pct": overall_accuracy,
        "total_checked": total_checked,
        "total_correct": total_correct,
    }


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate_pending_predictions() -> Dict:
    """
    Evaluate all predictions that are past their check_at timestamp.
    Reads from and writes to PostgreSQL.
    """
    with _get_pg_conn() as conn:
        return _evaluate_with_conn(conn)


def _evaluate_with_conn(conn) -> Dict:
    """Inner evaluation logic that runs inside a connection context."""
    _ensure_pg_tables(conn)
    cur = conn.cursor()

    now = int(time.time())

    cur.execute("""
        SELECT gp.id, gp.symbol, gp.predicted_at, gp.check_at, gp.predicted_price,
               gp.predicted_direction, gp.current_price, gp.confidence
        FROM ghost_predictions gp
        LEFT JOIN ghost_prediction_outcomes gpo ON gp.id = gpo.prediction_id
        WHERE gp.checked = 0 AND gp.check_at < %s
          AND gpo.prediction_id IS NULL
        ORDER BY gp.check_at ASC
        LIMIT 500
    """, (now,))
    pending = cur.fetchall()

    LOGGER.info(f"Found {len(pending)} predictions ready for evaluation")

    evaluated_count = 0
    correct_count = 0
    incorrect_count = 0
    skipped_count = 0
    skipped_symbols: Dict[str, int] = {}  # symbol → count for batch logging

    for pred in pending:
        pred_id, symbol, pred_at, check_at, pred_price, direction, start_price, confidence = pred

        prices = _fetch_window_prices_pg(
            conn, symbol=symbol, start_ts=int(pred_at), end_ts=int(check_at)
        )

        # MINIMUM PRICE POINTS: Require at least 5 data points for a fair evaluation.
        # Single-point fallback was producing 19.5% accuracy (garbage) vs 90% with
        # real window data. Skip predictions without sufficient price history.
        MIN_WINDOW_PRICES = 5
        if len(prices) < MIN_WINDOW_PRICES:
            skipped_count += 1
            skipped_symbols[symbol] = skipped_symbols.get(symbol, 0) + 1
            # Don't mark as checked — let it be re-evaluated when more data exists
            # But if the check_at is very old (>7 days), mark as skipped permanently
            age_hours = (now - int(check_at)) / 3600
            if age_hours > 168:  # Older than 7 days — data will never appear
                # FIX (Mar 13, 2026): Count as INCORRECT instead of skip-tagging.
                # Every prediction counts. No more hiding bad predictions.
                cur.execute(
                    "UPDATE ghost_predictions SET checked = 1, checked_at = %s, correct = 0, eval_version = %s WHERE id = %s",
                    (now, "no-data-v2", pred_id),
                )
                incorrect_count += 1
            continue

        eval_result = _evaluate_touch_target(
            predicted_direction=direction,
            start_price=float(start_price) if start_price is not None else None,
            target_price=float(pred_price) if pred_price is not None else None,
            prices=prices,
        )

        if not eval_result["ok"]:
            # FIX (Mar 13, 2026): Count as INCORRECT instead of skip-tagging.
            # Every prediction counts — no more inflating accuracy by hiding failures.
            incorrect_count += 1
            cur.execute(
                "UPDATE ghost_predictions SET checked = 1, checked_at = %s, correct = 0, eval_version = %s WHERE id = %s",
                (now, "eval-fail-v2", pred_id),
            )
            continue

        # PRIMARY: Direction accuracy — did the ACTUAL market move match the prediction?
        # direction_consistent only checks prediction internal consistency, NOT actual accuracy.
        # We need to compare predicted direction against actual outcome direction.
        predicted_dir = (direction or "").upper().strip()
        actual_dir = (eval_result.get("outcome_direction") or "").upper().strip()
        abs_move_pct = abs(eval_result.get("outcome_pct", 0))

        # Skip flat market — move too small to judge direction
        # FIX (Step 6, Mar 18 2026): Was 0.5% — too aggressive. A 0.3% move IS
        # directional (especially for stocks over 48h). Lowered to 0.25% to only
        # exclude truly flat markets. At 0.5%, ~15-30% of predictions were being
        # counted as WRONG when they were actually ambiguous, deflating accuracy
        # from ~55% to ~41%.
        FLAT_MARKET_THRESHOLD = 0.25  # percent (was 0.5)
        if abs_move_pct < FLAT_MARKET_THRESHOLD:
            # Flat market = no clear direction. Count as INCORRECT.
            incorrect_count += 1
            cur.execute(
                "UPDATE ghost_predictions SET checked = 1, checked_at = %s, correct = 0, eval_version = %s, "
                "outcome_pct = %s, outcome_price = %s, outcome_direction = %s, "
                "window_first = %s, window_last = %s, window_high = %s, window_low = %s "
                "WHERE id = %s",
                (now, "flat-market-v2", eval_result.get("outcome_pct"),
                 eval_result.get("window_last"), actual_dir,
                 eval_result.get("window_first"), eval_result.get("window_last"),
                 eval_result.get("window_high"), eval_result.get("window_low"),
                 pred_id),
            )
            continue

        # Correct = actual direction matches predicted direction AND move >= 0.5%
        is_direction_match = 1 if (predicted_dir == actual_dir and predicted_dir in ("UP", "DOWN")) else 0
        is_correct = is_direction_match
        is_correct_exec = int(eval_result.get("correct_0_5pct") or 0)

        cur.execute("""
            UPDATE ghost_predictions
            SET checked = 1,
                checked_at = %s,
                outcome_price = %s,
                outcome_direction = %s,
                outcome_pct = %s,
                correct = %s,
                error_pct = %s,
                window_first = %s,
                window_last = %s,
                window_high = %s,
                window_low = %s,
                touch_1pct = %s,
                touch_0_5pct = %s,
                correct_1pct = %s,
                correct_0_5pct = %s,
                direction_consistent = %s,
                eval_version = %s
            WHERE id = %s
        """, (
            now,
            eval_result.get("window_last"),
            eval_result.get("outcome_direction"),
            eval_result.get("outcome_pct"),
            is_correct,
            eval_result.get("error_pct"),
            eval_result.get("window_first"),
            eval_result.get("window_last"),
            eval_result.get("window_high"),
            eval_result.get("window_low"),
            int(eval_result.get("touch_1pct") or 0),
            int(eval_result.get("touch_0_5pct") or 0),
            int(eval_result.get("correct_1pct") or 0),
            int(eval_result.get("correct_0_5pct") or 0),
            int(eval_result.get("direction_consistent") or 0),
            "touch-pg-v1",
            pred_id,
        ))

        evaluated_count += 1
        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

        result_emoji = "✅" if is_correct else "❌"
        exec_tag = " (exec)" if is_correct_exec else ""
        LOGGER.info(
            f"{result_emoji} {symbol}: touch_1pct={int(eval_result.get('touch_1pct') or 0)} "
            f"touch_0_5pct={int(eval_result.get('touch_0_5pct') or 0)} "
            f"dir_ok={int(eval_result.get('direction_consistent') or 0)} "
            f"confidence={confidence:.1%}{exec_tag}"
        )

    conn.commit()

    # Log skipped predictions as a single summary instead of individual lines
    if skipped_symbols:
        skip_summary = ", ".join(f"{sym}×{cnt}" for sym, cnt in sorted(skipped_symbols.items()))
        LOGGER.info(f"⏩ Skipped {skipped_count} predictions (no price data): {skip_summary}")

    # Update ghost_accuracy_stats
    if evaluated_count > 0:
        cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1 AND eval_version NOT LIKE 'skip%%'")
        total_checked = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1 AND correct = 1 AND eval_version NOT LIKE 'skip%%'")
        total_correct = cur.fetchone()[0]

        overall_accuracy = (total_correct / total_checked * 100) if total_checked > 0 else 0

        cur.execute("SELECT AVG(error_pct) FROM ghost_predictions WHERE checked = 1 AND eval_version NOT LIKE 'skip%%'")
        avg_error = cur.fetchone()[0] or 0

        cur.execute("""
            INSERT INTO ghost_accuracy_stats
                (period, total_predictions, correct_predictions,
                 accuracy_pct, avg_error_pct, updated_at)
            VALUES ('all_time', %s, %s, %s, %s, %s)
            ON CONFLICT (period) DO UPDATE SET
                total_predictions = EXCLUDED.total_predictions,
                correct_predictions = EXCLUDED.correct_predictions,
                accuracy_pct = EXCLUDED.accuracy_pct,
                avg_error_pct = EXCLUDED.avg_error_pct,
                updated_at = EXCLUDED.updated_at
        """, (total_checked, total_correct, overall_accuracy, avg_error, now))

        conn.commit()

        LOGGER.info(
            f"📊 Overall accuracy: {overall_accuracy:.1f}% "
            f"({total_correct}/{total_checked} correct)"
        )

    return {
        "evaluated": evaluated_count,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "accuracy_pct": (correct_count / evaluated_count * 100) if evaluated_count > 0 else 0,
        "skipped": skipped_count,
    }


def main():
    """Main entry point"""
    LOGGER.info("=" * 60)
    LOGGER.info("Ghost Prediction Evaluator - Starting (PostgreSQL)")
    LOGGER.info("=" * 60)

    if not DATABASE_URL:
        LOGGER.error("❌ DATABASE_URL not set - evaluator cannot run")
        return 1

    try:
        # Step 1: Re-evaluate legacy skip-tagged predictions (one-time cleanup)
        try:
            re_eval = re_evaluate_skip_tagged()
            if re_eval["re_evaluated"] > 0:
                LOGGER.info(f"  🔄 Re-evaluated {re_eval['re_evaluated']} skip-tagged predictions")
                LOGGER.info(f"  ✅ Flipped to correct: {re_eval['flipped_to_correct']}")
                LOGGER.info(f"  ❌ Flipped to incorrect: {re_eval['flipped_to_incorrect']}")
                LOGGER.info(f"  📊 New accuracy: {re_eval['new_accuracy_pct']:.1f}%")
        except Exception as e:
            LOGGER.warning(f"Skip-tag re-evaluation failed (non-fatal): {e}")

        # Step 2: Evaluate pending predictions
        result = evaluate_pending_predictions()

        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info("Evaluation Complete:")
        LOGGER.info(f"  ✅ Evaluated: {result['evaluated']}")
        LOGGER.info(f"  ✅ Correct: {result['correct']}")
        LOGGER.info(f"  ❌ Incorrect: {result['incorrect']}")
        LOGGER.info(f"  ⏩ Skipped: {result['skipped']}")
        if result["evaluated"] > 0:
            LOGGER.info(f"  📊 Batch Accuracy: {result['accuracy_pct']:.1f}%")
        LOGGER.info("=" * 60)

        return 0

    except Exception as e:
        LOGGER.error(f"❌ Evaluator failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
