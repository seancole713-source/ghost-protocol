#!/usr/bin/env python3
"""
📊 GHOST PREDICTION ACCURACY TRACKER

Tracks every prediction Ghost makes and compares with actual outcomes.
Measures accuracy to verify Ghost's 85%+ target.

This is CRITICAL for validating Ghost's investment hunter capabilities.
"""

import logging
import os
import sqlite3
import time
from typing import Any

LOGGER = logging.getLogger("ghost.prediction_tracker")

# Database
WOLF_SQLITE_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")


def _ensure_prediction_tables():
    """Create prediction tracking tables if they don't exist."""
    conn = sqlite3.connect(WOLF_SQLITE_PATH)
    cur = conn.cursor()

    # Predictions table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ghost_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            predicted_at INTEGER NOT NULL,
            check_at INTEGER NOT NULL,
            predicted_price REAL,
            predicted_direction TEXT,
            predicted_pct REAL,
            confidence REAL,
            timeframe_hours INTEGER,
            reasons TEXT,
            current_price REAL,
            outcome_price REAL,
            outcome_direction TEXT,
            outcome_pct REAL,
            correct INTEGER,
            checked INTEGER DEFAULT 0,
            checked_at INTEGER,
            error_pct REAL,
            features_json TEXT,
            UNIQUE(symbol, predicted_at)
        )
    """
    )

    # Accuracy stats table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ghost_accuracy_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period TEXT NOT NULL,
            total_predictions INTEGER,
            correct_predictions INTEGER,
            accuracy_pct REAL,
            avg_error_pct REAL,
            best_symbol TEXT,
            worst_symbol TEXT,
            updated_at INTEGER,
            UNIQUE(period)
        )
    """
    )

    conn.commit()
    conn.close()


# Initialize tables on import
_ensure_prediction_tables()


def log_prediction(
    symbol: str,
    predicted_price: float,
    predicted_direction: str,
    predicted_pct: float,
    confidence: float,
    timeframe_hours: int,
    reasons: list[str],
    current_price: float,
) -> int:
    """
    Log a new prediction.
    
    Args:
        symbol: Stock/crypto symbol
        predicted_price: Predicted future price
        predicted_direction: UP/DOWN/FLAT
        predicted_pct: Predicted % change
        confidence: AI confidence (0-1)
        timeframe_hours: Prediction window (2-48 hours)
        reasons: List of reasoning strings
        current_price: Current price at prediction time
        
    Returns:
        prediction_id
    """
    try:
        predicted_at = int(time.time())
        check_at = predicted_at + (timeframe_hours * 3600)

        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()

        cur.execute(
            """
            INSERT OR REPLACE INTO ghost_predictions
            (symbol, predicted_at, check_at, predicted_price, predicted_direction,
             predicted_pct, confidence, timeframe_hours, reasons, current_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                symbol,
                predicted_at,
                check_at,
                predicted_price,
                predicted_direction,
                predicted_pct,
                confidence,
                timeframe_hours,
                "|".join(reasons),
                current_price,
            ),
        )

        prediction_id = cur.lastrowid
        conn.commit()
        conn.close()

        LOGGER.info(
            f"📊 Logged prediction #{prediction_id}: {symbol} "
            f"{predicted_direction} {predicted_pct:+.2f}% "
            f"(confidence: {confidence:.0%}, check in {timeframe_hours}h)"
        )

        return prediction_id

    except Exception as e:
        LOGGER.error(f"Failed to log prediction: {e}")
        return -1


def check_predictions_due() -> list[dict[str, Any]]:
    """
    Check predictions that are now due for accuracy verification.
    
    Returns:
        List of predictions that need checking
    """
    try:
        now = int(time.time())

        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            SELECT * FROM ghost_predictions
            WHERE checked = 0 AND check_at <= ?
            ORDER BY check_at ASC
            LIMIT 50
        """,
            (now,),
        )

        rows = cur.fetchall()
        conn.close()

        predictions = []
        for row in rows:
            predictions.append(
                {
                    "id": row["id"],
                    "symbol": row["symbol"],
                    "predicted_at": row["predicted_at"],
                    "check_at": row["check_at"],
                    "predicted_price": row["predicted_price"],
                    "predicted_direction": row["predicted_direction"],
                    "predicted_pct": row["predicted_pct"],
                    "confidence": row["confidence"],
                    "timeframe_hours": row["timeframe_hours"],
                    "current_price": row["current_price"],
                }
            )

        if predictions:
            LOGGER.info(f"📊 Found {len(predictions)} predictions due for checking")

        return predictions

    except Exception as e:
        LOGGER.error(f"Failed to check predictions: {e}")
        return []


async def verify_prediction(prediction: dict[str, Any]) -> bool:
    """
    Verify a prediction by fetching actual outcome price.
    
    Args:
        prediction: Prediction dict from check_predictions_due()
        
    Returns:
        True if verified successfully
    """
    try:
        symbol = prediction["symbol"]
        predicted_price = prediction["predicted_price"]
        predicted_direction = prediction["predicted_direction"]
        predicted_pct = prediction["predicted_pct"]
        current_price = prediction["current_price"]

        # Fetch actual outcome price
        try:
            from core.price import get_price

            outcome_price = await get_price(symbol)

            if outcome_price is None:
                LOGGER.warning(f"Could not get outcome price for {symbol}")
                return False

        except Exception as e:
            LOGGER.error(f"Failed to fetch outcome price for {symbol}: {e}")
            return False

        # Calculate actual outcome
        outcome_pct = ((outcome_price - current_price) / current_price) * 100
        outcome_direction = "UP" if outcome_pct > 0 else ("DOWN" if outcome_pct < 0 else "FLAT")

        # Determine correctness
        direction_correct = predicted_direction == outcome_direction
        error_pct = abs(predicted_pct - outcome_pct)

        # Mark as correct if direction matches
        correct = 1 if direction_correct else 0

        # Update database
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE ghost_predictions
            SET outcome_price = ?,
                outcome_direction = ?,
                outcome_pct = ?,
                correct = ?,
                checked = 1,
                checked_at = ?,
                error_pct = ?
            WHERE id = ?
        """,
            (
                outcome_price,
                outcome_direction,
                outcome_pct,
                correct,
                int(time.time()),
                error_pct,
                prediction["id"],
            ),
        )

        conn.commit()
        conn.close()

        result_emoji = "✅" if correct else "❌"
        LOGGER.info(
            f"{result_emoji} Verified prediction #{prediction['id']}: {symbol} "
            f"predicted {predicted_direction} {predicted_pct:+.2f}%, "
            f"actual {outcome_direction} {outcome_pct:+.2f}% (error: {error_pct:.2f}%)"
        )

        return True

    except Exception as e:
        LOGGER.error(f"Failed to verify prediction: {e}")
        return False


def calculate_accuracy(period: str = "all") -> dict[str, Any]:
    """
    Calculate Ghost's prediction accuracy.
    
    Args:
        period: 'all', '24h', '7d', '30d'
        
    Returns:
        Accuracy statistics
    """
    try:
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Time filter
        time_filter = ""
        if period == "24h":
            cutoff = int(time.time()) - (24 * 3600)
            time_filter = f"AND predicted_at >= {cutoff}"
        elif period == "7d":
            cutoff = int(time.time()) - (7 * 24 * 3600)
            time_filter = f"AND predicted_at >= {cutoff}"
        elif period == "30d":
            cutoff = int(time.time()) - (30 * 24 * 3600)
            time_filter = f"AND predicted_at >= {cutoff}"

        # Get checked predictions
        cur.execute(
            f"""
            SELECT * FROM ghost_predictions
            WHERE checked = 1 AND confidence >= 0.10 {time_filter}
            ORDER BY predicted_at DESC
        """
        )

        rows = cur.fetchall()
        conn.close()

        if not rows:
            return {
                "period": period,
                "total_predictions": 0,
                "correct_predictions": 0,
                "accuracy_pct": 0.0,
                "avg_error_pct": 0.0,
                "predictions": [],
            }

        total = len(rows)
        correct = sum(1 for row in rows if row["correct"] == 1)
        accuracy_pct = (correct / total) * 100 if total > 0 else 0

        errors = [row["error_pct"] for row in rows if row["error_pct"] is not None]
        avg_error_pct = sum(errors) / len(errors) if errors else 0

        # Convert rows to dicts
        predictions = []
        for row in rows:
            predictions.append(
                {
                    "symbol": row["symbol"],
                    "predicted_at": row["predicted_at"],
                    "predicted_direction": row["predicted_direction"],
                    "predicted_pct": row["predicted_pct"],
                    "outcome_direction": row["outcome_direction"],
                    "outcome_pct": row["outcome_pct"],
                    "correct": bool(row["correct"]),
                    "error_pct": row["error_pct"],
                    "confidence": row["confidence"],
                }
            )

        result = {
            "period": period,
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy_pct": accuracy_pct,
            "avg_error_pct": avg_error_pct,
            "predictions": predictions,
        }

        # Store stats
        _store_accuracy_stats(period, result)

        LOGGER.info(
            f"📊 Accuracy ({period}): {accuracy_pct:.1f}% "
            f"({correct}/{total} correct, avg error: {avg_error_pct:.2f}%)"
        )

        return result

    except Exception as e:
        LOGGER.error(f"Failed to calculate accuracy: {e}")
        return {
            "period": period,
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy_pct": 0.0,
            "avg_error_pct": 0.0,
            "predictions": [],
            "error": str(e),
        }


def _store_accuracy_stats(period: str, stats: dict[str, Any]):
    """Store accuracy stats in database for historical tracking."""
    try:
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()

        cur.execute(
            """
            INSERT OR REPLACE INTO ghost_accuracy_stats
            (period, total_predictions, correct_predictions, accuracy_pct,
             avg_error_pct, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                period,
                stats["total_predictions"],
                stats["correct_predictions"],
                stats["accuracy_pct"],
                stats["avg_error_pct"],
                int(time.time()),
            ),
        )

        conn.commit()
        conn.close()

    except Exception as e:
        LOGGER.error(f"Failed to store accuracy stats: {e}")


async def accuracy_check_loop():
    """
    Background loop that checks predictions when they become due.
    Runs every 5 minutes.
    """
    LOGGER.info("📊 Prediction accuracy checker started")

    while True:
        try:
            # Find predictions due for checking
            predictions_due = check_predictions_due()

            # Verify each one
            for prediction in predictions_due:
                await verify_prediction(prediction)
                # Small delay between checks
                await asyncio.sleep(1)

            # Calculate and log overall accuracy
            if predictions_due:
                calculate_accuracy("all")
                calculate_accuracy("7d")
                calculate_accuracy("24h")

            # Wait 5 minutes before next check
            await asyncio.sleep(300)

        except Exception as e:
            LOGGER.error(f"Error in accuracy check loop: {e}", exc_info=True)
            await asyncio.sleep(600)  # Wait longer after error


async def start_accuracy_tracker():
    """Start the accuracy tracker as a background task."""
    import asyncio

    await accuracy_check_loop()


if __name__ == "__main__":
    # Test standalone
    import asyncio

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    print("📊 Starting Ghost Accuracy Tracker (standalone mode)")
    print()

    asyncio.run(accuracy_check_loop())
