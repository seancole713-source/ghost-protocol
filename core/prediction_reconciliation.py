"""
Prediction Outcome Reconciliation Job
======================================

Automated job that runs every 4 hours to:
1. Fetch all predictions with horizon windows that have closed
2. Get actual prices for those symbols
3. Calculate directional accuracy (UP/DOWN/FLAT correctness)
4. Update prediction outcomes in database
5. Calculate rolling accuracy metrics

This provides the "learning feedback loop" for Ghost.

Usage:
    # Run once (manual)
    python -m core.prediction_reconciliation

    # Run as background worker (auto every 4h)
    from core.prediction_reconciliation import start_reconciliation_worker
    start_reconciliation_worker()

Author: Ghost AI
Date: November 21, 2025
"""

import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Database paths
PREDICTIONS_DB = Path(__file__).parent.parent / "wolf.db"
OUTCOMES_DB = Path(__file__).parent.parent / "data" / "prediction_outcomes.db"


class PredictionReconciliation:
    """Reconciles predictions with actual outcomes."""

    def __init__(self):
        """Initialize reconciliation engine"""
        self._init_outcomes_db()

    def _init_outcomes_db(self):
        """Create outcomes tracking database"""
        OUTCOMES_DB.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(OUTCOMES_DB)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_outcomes (
                    prediction_id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    predicted_at REAL NOT NULL,
                    horizon_hours INTEGER NOT NULL,
                    direction_predicted TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    price_at_prediction REAL NOT NULL,
                    price_at_outcome REAL,
                    actual_change_pct REAL,
                    direction_actual TEXT,
                    direction_correct INTEGER,
                    outcome_timestamp REAL,
                    reconciled_at REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Accuracy metrics table (rolling stats)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    period_days INTEGER NOT NULL,
                    total_predictions INTEGER NOT NULL,
                    correct_predictions INTEGER NOT NULL,
                    accuracy_pct REAL NOT NULL,
                    avg_confidence REAL NOT NULL,
                    calculated_at REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_symbol
                ON prediction_outcomes(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_reconciled
                ON prediction_outcomes(reconciled_at)
            """)

            conn.commit()
            logger.info(f"Outcomes database initialized: {OUTCOMES_DB}")

    def reconcile_pending_predictions(self) -> dict[str, Any]:
        """
        Find predictions with closed time windows and reconcile outcomes.
        
        Returns:
            {
                "reconciled": int,
                "skipped": int,
                "errors": list[str]
            }
        """
        start_time = time.time()
        reconciled = 0
        skipped = 0
        errors = []

        try:
            # Get predictions from wolf.db
            pending = self._fetch_pending_predictions()
            logger.info(f"Found {len(pending)} predictions to reconcile")

            for pred in pending:
                try:
                    if self._reconcile_prediction(pred):
                        reconciled += 1
                    else:
                        skipped += 1
                except Exception as e:
                    logger.error(f"Failed to reconcile prediction {pred.get('id')}: {e}")
                    errors.append(f"Prediction {pred.get('id')}: {str(e)}")

        except Exception as e:
            logger.error(f"Reconciliation job failed: {e}")
            errors.append(f"Job failure: {str(e)}")

        execution_time = time.time() - start_time

        result = {
            "reconciled": reconciled,
            "skipped": skipped,
            "errors": errors,
            "execution_time_s": round(execution_time, 2),
            "timestamp": time.time(),
        }

        logger.info(
            f"Reconciliation complete: {reconciled} reconciled, "
            f"{skipped} skipped in {execution_time:.1f}s"
        )

        return result

    def _fetch_pending_predictions(self) -> list[dict[str, Any]]:
        """Fetch predictions that need reconciliation from PostgreSQL"""
        predictions = []

        try:
            # Use PostgreSQL prediction store instead of SQLite
            from core.prediction_store import get_prediction_store
            import os
            
            store = get_prediction_store()
            now = time.time()
            
            # Check if using PostgreSQL (production) or SQLite (local)
            is_postgres = os.getenv("DATABASE_URL", "").startswith("postgresql")
            
            if is_postgres and hasattr(store, 'engine'):
                # PostgreSQL query
                from sqlalchemy import text
                logger.info("Fetching pending predictions from PostgreSQL...")
                
                with store.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT 
                            id,
                            symbol,
                            run_at,
                            horizon_h,
                            direction,
                            confidence,
                            features_json
                        FROM ghost_predictions
                        WHERE run_at < :now - (horizon_h * 3600)
                        ORDER BY run_at DESC
                        LIMIT 100
                    """), {"now": now})
                    
                    rows = result.fetchall()
                    logger.info(f"Found {len(rows)} predictions ready for reconciliation in PostgreSQL")
                    
                    for row in rows:
                        import json
                        features = json.loads(row[6]) if row[6] else {}
                        
                        predictions.append({
                            "id": row[0],
                            "symbol": row[1],
                            "run_at": row[2],
                            "horizon_h": row[3],
                            "direction": row[4],
                            "confidence": row[5],
                            "price_at_prediction": features.get("current_price", 0)
                        })
            else:
                # Fallback to SQLite for local development
                logger.info("Fetching pending predictions from SQLite (local)...")
                with sqlite3.connect(str(PREDICTIONS_DB)) as conn:
                    rows = conn.execute("""
                        SELECT 
                            id,
                            symbol,
                            run_at,
                            horizon_h,
                            direction,
                            confidence,
                            features
                        FROM ghost_predictions
                        WHERE run_at < ? - (horizon_h * 3600)
                        ORDER BY run_at DESC
                        LIMIT 100
                    """, (now,)).fetchall()
                    
                    logger.info(f"Found {len(rows)} predictions ready for reconciliation in SQLite")

                    for row in rows:
                        import json
                        features = json.loads(row[6]) if row[6] else {}
                        
                        predictions.append({
                            "id": row[0],
                            "symbol": row[1],
                            "run_at": row[2],
                            "horizon_h": row[3],
                            "direction": row[4],
                            "confidence": row[5],
                            "price_at_prediction": features.get("current_price", 0)
                        })

        except Exception as e:
            logger.error(f"Failed to fetch pending predictions: {e}", exc_info=True)

        return predictions

    def _reconcile_prediction(self, pred: dict[str, Any]) -> bool:
        """
        Reconcile a single prediction with actual outcome.
        
        Returns:
            True if reconciled successfully, False if skipped
        """
        # Check if already reconciled
        with sqlite3.connect(str(OUTCOMES_DB)) as conn:
            existing = conn.execute(
                "SELECT prediction_id FROM prediction_outcomes WHERE prediction_id = ?",
                (pred["id"],)
            ).fetchone()

            if existing:
                return False  # Already reconciled

        # Get actual price at outcome time
        symbol = pred["symbol"]
        outcome_time = pred["run_at"] + (pred["horizon_h"] * 3600)

        try:
            # Import wolf_app price fetcher
            import os
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from wolf_app import _get_price_quorum

            price_data = _get_price_quorum(symbol, "stock")
            if not price_data or not price_data.get("price"):
                logger.warning(f"Cannot get current price for {symbol}, skipping")
                return False

            actual_price = float(price_data["price"])
            price_at_prediction = pred["price_at_prediction"]

            if price_at_prediction <= 0:
                logger.warning(f"Invalid price_at_prediction for {symbol}, skipping")
                return False

            # Calculate actual change
            actual_change_pct = ((actual_price - price_at_prediction) / price_at_prediction) * 100

            # Determine actual direction (threshold: ±1%)
            if actual_change_pct > 1.0:
                direction_actual = "UP"
            elif actual_change_pct < -1.0:
                direction_actual = "DOWN"
            else:
                direction_actual = "FLAT"

            # Check if direction was correct
            direction_correct = 1 if direction_actual == pred["direction"] else 0

            # Store outcome
            with sqlite3.connect(str(OUTCOMES_DB)) as conn:
                conn.execute("""
                    INSERT INTO prediction_outcomes (
                        prediction_id, symbol, predicted_at, horizon_hours,
                        direction_predicted, confidence, price_at_prediction,
                        price_at_outcome, actual_change_pct, direction_actual,
                        direction_correct, outcome_timestamp, reconciled_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pred["id"],
                    symbol,
                    pred["run_at"],
                    pred["horizon_h"],
                    pred["direction"],
                    pred["confidence"],
                    price_at_prediction,
                    actual_price,
                    actual_change_pct,
                    direction_actual,
                    direction_correct,
                    outcome_time,
                    time.time()
                ))
                conn.commit()

            logger.info(
                f"Reconciled {symbol}: predicted {pred['direction']}, "
                f"actual {direction_actual} ({actual_change_pct:+.2f}%) - "
                f"{'✓' if direction_correct else '✗'}"
            )

            return True

        except Exception as e:
            logger.error(f"Reconciliation failed for {symbol}: {e}")
            return False

    def calculate_accuracy_metrics(self, symbol: str | None = None, period_days: int = 30) -> dict[str, Any]:
        """
        Calculate rolling accuracy metrics.
        
        Args:
            symbol: Symbol to calculate for (None = all)
            period_days: Lookback period
        
        Returns:
            Accuracy stats dict
        """
        cutoff_time = time.time() - (period_days * 86400)

        with sqlite3.connect(str(OUTCOMES_DB)) as conn:
            if symbol:
                rows = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(direction_correct) as correct,
                        AVG(confidence) as avg_confidence
                    FROM prediction_outcomes
                    WHERE symbol = ? AND reconciled_at >= ?
                """, (symbol, cutoff_time)).fetchone()
            else:
                rows = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(direction_correct) as correct,
                        AVG(confidence) as avg_confidence
                    FROM prediction_outcomes
                    WHERE reconciled_at >= ?
                """, (cutoff_time,)).fetchone()

            if not rows or rows[0] == 0:
                return {
                    "ok": False,
                    "error": "No reconciled predictions found",
                    "symbol": symbol,
                    "period_days": period_days,
                }

            total = rows[0]
            correct = rows[1] or 0
            avg_confidence = rows[2] or 0.0

            accuracy_pct = (correct / total) * 100 if total > 0 else 0.0

            # Store in metrics table
            conn.execute("""
                INSERT INTO accuracy_metrics (
                    symbol, period_days, total_predictions, correct_predictions,
                    accuracy_pct, avg_confidence, calculated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol or "ALL",
                period_days,
                total,
                correct,
                accuracy_pct,
                avg_confidence,
                time.time()
            ))
            conn.commit()

        return {
            "ok": True,
            "symbol": symbol or "ALL",
            "period_days": period_days,
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy_pct": round(accuracy_pct, 2),
            "avg_confidence": round(avg_confidence, 2),
            "timestamp": time.time(),
        }


# Singleton instance
_reconciliation = None


def get_reconciliation() -> PredictionReconciliation:
    """Get singleton reconciliation instance"""
    global _reconciliation
    if _reconciliation is None:
        _reconciliation = PredictionReconciliation()
    return _reconciliation


def reconcile_predictions() -> dict[str, Any]:
    """Run reconciliation job (convenience function)"""
    return get_reconciliation().reconcile_pending_predictions()


def start_reconciliation_worker(interval_hours: int = 4):
    """
    Start background worker that reconciles predictions every N hours.
    
    Args:
        interval_hours: Run interval (default: 4 hours)
    """
    def worker():
        logger.info(f"Reconciliation worker started (interval: {interval_hours}h)")
        while True:
            try:
                result = reconcile_predictions()
                logger.info(f"Reconciliation cycle complete: {result}")
            except Exception as e:
                logger.error(f"Reconciliation worker error: {e}")
            
            time.sleep(interval_hours * 3600)

    thread = threading.Thread(target=worker, daemon=True, name="prediction-reconciliation")
    thread.start()
    logger.info("Reconciliation worker thread started")


if __name__ == "__main__":
    # Run reconciliation manually
    logging.basicConfig(level=logging.INFO)
    result = reconcile_predictions()
    print(result)
