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

DB_PATH = "data/wolf.db"


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
    conn = sqlite3.connect(DB_PATH)
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
    
    for pred in pending:
        pred_id, symbol, pred_at, check_at, pred_price, direction, start_price, confidence = pred
        
        # Fetch outcome price
        outcome_price = get_current_price(symbol)
        
        if not outcome_price or outcome_price <= 0:
            LOGGER.warning(f"⏩ Skipping {symbol} (prediction {pred_id}): Could not fetch outcome price")
            skipped_count += 1
            continue
        
        # Calculate price change percentage
        price_change_pct = ((outcome_price - start_price) / start_price) * 100
        
        # Determine actual direction based on 1% threshold
        if price_change_pct > 1.0:
            actual_direction = "UP"
        elif price_change_pct < -1.0:
            actual_direction = "DOWN"
        else:
            actual_direction = "FLAT"
        
        # Check if prediction was correct
        is_correct = 1 if direction == actual_direction else 0
        
        # Calculate error percentage
        error_pct = abs(outcome_price - pred_price) / start_price * 100
        
        # Update database
        cur.execute("""
            UPDATE ghost_predictions
            SET checked = 1,
                checked_at = ?,
                outcome_price = ?,
                outcome_direction = ?,
                outcome_pct = ?,
                correct = ?,
                error_pct = ?
            WHERE id = ?
        """, (
            now,
            outcome_price,
            actual_direction,
            price_change_pct,
            is_correct,
            error_pct,
            pred_id
        ))
        
        evaluated_count += 1
        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1
        
        # Log result
        result_emoji = "✅" if is_correct else "❌"
        LOGGER.info(
            f"{result_emoji} {symbol}: "
            f"predicted={direction}, actual={actual_direction}, "
            f"change={price_change_pct:+.2f}%, confidence={confidence:.1%}"
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
