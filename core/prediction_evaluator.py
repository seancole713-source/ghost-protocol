#!/usr/bin/env python3
"""
Ghost Prediction Evaluator

Checks predictions that are ready for evaluation (48h elapsed),
fetches outcome prices, determines if prediction was correct,
and updates the ghost_predictions table.

Run this as a cron job every hour:
0 * * * * cd /app && python3 core/prediction_evaluator.py >> /tmp/evaluator.log 2>&1
"""

import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOGGER = logging.getLogger(__name__)

DB_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")


def _ensure_touch_columns(conn: sqlite3.Connection) -> None:
    """Ensure wolf.db ghost_predictions has columns needed for touch-target evaluation."""
    cur = conn.cursor()
    cols = [r[1] for r in cur.execute("PRAGMA table_info(ghost_predictions)").fetchall()]

    def _add(col: str, ddl: str) -> None:
        if col in cols:
            return
        cur.execute(f"ALTER TABLE ghost_predictions ADD COLUMN {ddl}")

    _add("window_first", "window_first REAL")
    _add("window_last", "window_last REAL")
    _add("window_high", "window_high REAL")
    _add("window_low", "window_low REAL")
    _add("target_price", "target_price REAL")
    _add("stage5_ok", "stage5_ok INTEGER")
    _add("stage6_ok", "stage6_ok INTEGER")
    _add("gate", "gate TEXT")
    _add("touch_calibrated_1pct", "touch_calibrated_1pct REAL")
    _add("touch_calibrated_0_5pct", "touch_calibrated_0_5pct REAL")
    _add("touch_calibration_samples", "touch_calibration_samples INTEGER")
    _add("touch_conf_band", "touch_conf_band TEXT")
    _add("touch_1pct", "touch_1pct INTEGER")
    _add("touch_0_5pct", "touch_0_5pct INTEGER")
    _add("correct_1pct", "correct_1pct INTEGER")
    _add("correct_0_5pct", "correct_0_5pct INTEGER")
    _add("direction_consistent", "direction_consistent INTEGER")
    _add("eval_version", "eval_version TEXT")
    # Keep schema compatible with prediction logging in wolf_app.py
    _add("features_json", "features_json TEXT")

    conn.commit()


def get_current_price(symbol: str) -> Optional[float]:
    """
    Fetch current price for a symbol using Ghost's price quorum system.
    
    Args:
        symbol: Trading symbol (e.g., "AAPL", "BTC")
    
    Returns:
        Current price as float, or None if failed
    """
    try:
        # Import Ghost's price fetching logic
        from wolf_app import _get_price_quorum, HUNTER_CRYPTO_SYMBOLS
        
        # Determine if crypto or stock
        is_crypto = symbol in HUNTER_CRYPTO_SYMBOLS
        
        if is_crypto:
            # Use async crypto price quorum (need to run in event loop)
            import asyncio
            from core.crypto.crypto_providers import get_crypto_price_quorum
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                crypto_data = loop.run_until_complete(get_crypto_price_quorum(symbol, use_cache=False))
                if crypto_data and crypto_data.get("price"):
                    return float(crypto_data["price"])
            finally:
                loop.close()
        else:
            # Use stock price quorum
            price_data = _get_price_quorum(symbol, "stock")
            if price_data and price_data.get("price"):
                return float(price_data["price"])
        
        return None
    
    except Exception as e:
        LOGGER.error(f"Failed to fetch price for {symbol}: {e}")
        return None


def evaluate_pending_predictions() -> Dict:
    """
    Evaluate all predictions that are past their check_at timestamp.
    
    Returns:
        Dict with evaluation stats: {
            "evaluated": int,
            "correct": int,
            "incorrect": int,
            "accuracy_pct": float,
            "skipped": int
        }
    """
    # Ensure base tables exist (fresh DB safe)
    try:
        from core.prediction_tracker import _ensure_prediction_tables
        _ensure_prediction_tables()
    except Exception:
        pass

    conn = sqlite3.connect(DB_PATH)
    _ensure_touch_columns(conn)
    cur = conn.cursor()
    
    now = int(time.time())
    
    # Find predictions ready for evaluation
    pending = cur.execute("""
        SELECT id, symbol, predicted_at, check_at, predicted_price, 
               predicted_direction, current_price, confidence
        FROM ghost_predictions
        WHERE checked = 0 AND check_at < ?
        ORDER BY check_at ASC
        LIMIT 100
    """, (now,)).fetchall()
    
    LOGGER.info(f"Found {len(pending)} predictions ready for evaluation")
    
    evaluated_count = 0
    correct_count = 0
    incorrect_count = 0
    skipped_count = 0
    
    from core.target_touch_evaluator import evaluate_prediction_row

    for pred in pending:
        pred_id, symbol, pred_at, check_at, pred_price, direction, start_price, confidence = pred

        eval_result = evaluate_prediction_row(
            conn,
            symbol=symbol,
            predicted_at=int(pred_at),
            check_at=int(check_at),
            predicted_price=float(pred_price) if pred_price is not None else None,
            predicted_direction=direction,
            current_price=float(start_price) if start_price is not None else None,
        )

        if not eval_result.ok:
            LOGGER.warning(
                f"⏩ Skipping {symbol} (prediction {pred_id}): {eval_result.reason}"
            )
            skipped_count += 1
            continue

        is_correct = int(eval_result.correct_1pct or 0)
        is_correct_exec = int(eval_result.correct_0_5pct or 0)

        # Update database
        cur.execute(
            """
            UPDATE ghost_predictions
            SET checked = 1,
                checked_at = ?,
                outcome_price = ?,
                outcome_direction = ?,
                outcome_pct = ?,
                correct = ?,
                error_pct = ?,
                window_first = ?,
                window_last = ?,
                window_high = ?,
                window_low = ?,
                touch_1pct = ?,
                touch_0_5pct = ?,
                correct_1pct = ?,
                correct_0_5pct = ?,
                direction_consistent = ?,
                eval_version = ?
            WHERE id = ?
            """,
            (
                now,
                float(eval_result.window_last) if eval_result.window_last is not None else None,
                eval_result.outcome_direction,
                float(eval_result.outcome_pct) if eval_result.outcome_pct is not None else None,
                is_correct,
                float(eval_result.error_pct) if eval_result.error_pct is not None else None,
                float(eval_result.window_first) if eval_result.window_first is not None else None,
                float(eval_result.window_last) if eval_result.window_last is not None else None,
                float(eval_result.window_high) if eval_result.window_high is not None else None,
                float(eval_result.window_low) if eval_result.window_low is not None else None,
                int(eval_result.touch_1pct or 0),
                int(eval_result.touch_0_5pct or 0),
                int(eval_result.correct_1pct or 0),
                int(eval_result.correct_0_5pct or 0),
                int(eval_result.direction_consistent or 0),
                "touch-v1",
                pred_id,
            ),
        )

        evaluated_count += 1
        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

        result_emoji = "✅" if is_correct else "❌"
        exec_tag = " (exec)" if is_correct_exec else ""
        LOGGER.info(
            f"{result_emoji} {symbol}: touch_1pct={int(eval_result.touch_1pct or 0)} "
            f"touch_0_5pct={int(eval_result.touch_0_5pct or 0)} dir_ok={int(eval_result.direction_consistent or 0)} "
            f"confidence={confidence:.1%}{exec_tag}"
        )
    
    conn.commit()
    
    # Update ghost_accuracy_stats table
    if evaluated_count > 0:
        accuracy_pct = (correct_count / evaluated_count) * 100
        
        # Calculate overall stats (all time)
        total_checked = cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1").fetchone()[0]
        total_correct = cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1 AND correct = 1").fetchone()[0]
        overall_accuracy = (total_correct / total_checked * 100) if total_checked > 0 else 0
        avg_error = cur.execute("SELECT AVG(error_pct) FROM ghost_predictions WHERE checked = 1").fetchone()[0] or 0
        
        # Upsert accuracy stats
        cur.execute("""
            INSERT OR REPLACE INTO ghost_accuracy_stats (
                id, period, total_predictions, correct_predictions, 
                accuracy_pct, avg_error_pct, updated_at
            ) VALUES (1, 'all_time', ?, ?, ?, ?, ?)
        """, (
            total_checked,
            total_correct,
            overall_accuracy,
            avg_error,
            now
        ))
        
        conn.commit()
        
        LOGGER.info(f"📊 Overall accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_checked} correct)")
    
    conn.close()
    
    return {
        "evaluated": evaluated_count,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "accuracy_pct": (correct_count / evaluated_count * 100) if evaluated_count > 0 else 0,
        "skipped": skipped_count
    }


def main():
    """Main entry point"""
    LOGGER.info("=" * 60)
    LOGGER.info("Ghost Prediction Evaluator - Starting")
    LOGGER.info("=" * 60)
    
    try:
        result = evaluate_pending_predictions()
        
        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info(f"Evaluation Complete:")
        LOGGER.info(f"  ✅ Evaluated: {result['evaluated']}")
        LOGGER.info(f"  ✅ Correct: {result['correct']}")
        LOGGER.info(f"  ❌ Incorrect: {result['incorrect']}")
        LOGGER.info(f"  ⏩ Skipped: {result['skipped']}")
        if result['evaluated'] > 0:
            LOGGER.info(f"  📊 Batch Accuracy: {result['accuracy_pct']:.1f}%")
        LOGGER.info("=" * 60)
        
        return 0
    
    except Exception as e:
        LOGGER.error(f"❌ Evaluator failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
