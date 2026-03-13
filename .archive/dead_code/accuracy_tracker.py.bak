"""
GHOST Stage 2: Accuracy Tracker
================================
Tracks forecast accuracy by comparing predictions vs actual prices.

Features:
- Store forecasts with timestamp, symbol, predicted price, confidence
- Compare predictions to actual prices after time window
- Calculate MAP (Mean Absolute Percentage Error)
- Calculate RMSE (Root Mean Square Error)
- Track bias (over-prediction vs under-prediction)
- Historical accuracy trends

Intelligence Level: 8 → 9 (Self-Evaluation System)

Author: Ghost AI
Date: 2025-10-05
"""

import json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "forecast_accuracy.db"


class AccuracyTracker:
    """
    Tracks forecast accuracy and model performance.

    Workflow:
    1. record_forecast() - Store prediction when made
    2. update_actual() - Update with actual price (manual or auto)
    3. calculate_metrics() - Compute MAP, RMSE, bias
    4. get_accuracy_report() - Return performance summary
    """

    def __init__(self, db_path: str | None = None):
        """Initialize accuracy tracker with database."""
        self.db_path = db_path or str(DB_PATH)
        self._init_db()

    def _init_db(self):
        """Create database tables if they don't exist."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    forecast_price REAL NOT NULL,
                    forecast_horizon_hours INTEGER NOT NULL,
                    confidence REAL,
                    actual_price REAL,
                    actual_timestamp REAL,
                    absolute_error REAL,
                    percentage_error REAL,
                    squared_error REAL,
                    model_version TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Index for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_forecasts_symbol
                ON forecasts(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_forecasts_timestamp
                ON forecasts(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_forecasts_actual
                ON forecasts(actual_price)
            """)

            conn.commit()
            logger.info(f"Accuracy tracker initialized: {self.db_path}")

    def record_forecast(
        self,
        symbol: str,
        forecast_price: float,
        forecast_horizon_hours: int = 24,
        confidence: float | None = None,
        model_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Record a new forecast.

        Args:
            symbol: Stock ticker (e.g., 'WOLF')
            forecast_price: Predicted price
            forecast_horizon_hours: How far ahead (default 24h)
            confidence: Model confidence (0-1)
            model_version: Model identifier for tracking
            metadata: Additional context (dict)

        Returns:
            Forecast ID
        """
        timestamp = time.time()
        metadata_json = json.dumps(metadata) if metadata else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO forecasts (
                    timestamp, symbol, forecast_price, forecast_horizon_hours,
                    confidence, model_version, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    symbol,
                    forecast_price,
                    forecast_horizon_hours,
                    confidence,
                    model_version,
                    metadata_json,
                ),
            )
            forecast_id = cursor.lastrowid
            assert forecast_id is not None, "Failed to get forecast ID from database"
            conn.commit()

        logger.info(
            f"Forecast recorded: {symbol} @ ${forecast_price:.2f} "
            f"(horizon={forecast_horizon_hours}h, id={forecast_id})"
        )
        return forecast_id

    def update_actual(
        self, forecast_id: int, actual_price: float, actual_timestamp: float | None = None
    ) -> bool:
        """
        Update forecast with actual price and calculate errors.

        Args:
            forecast_id: ID from record_forecast()
            actual_price: Observed price
            actual_timestamp: When price was observed (default: now)

        Returns:
            True if updated successfully
        """
        if actual_timestamp is None:
            actual_timestamp = time.time()

        with sqlite3.connect(self.db_path) as conn:
            # Get forecast
            row = conn.execute(
                "SELECT forecast_price FROM forecasts WHERE id = ?", (forecast_id,)
            ).fetchone()

            if not row:
                logger.warning(f"Forecast {forecast_id} not found")
                return False

            forecast_price = row[0]

            # Calculate errors
            absolute_error = abs(actual_price - forecast_price)
            percentage_error = (absolute_error / actual_price * 100.0) if actual_price > 0 else 0.0
            squared_error = (actual_price - forecast_price) ** 2

            # Update record
            conn.execute(
                """
                UPDATE forecasts SET
                    actual_price = ?,
                    actual_timestamp = ?,
                    absolute_error = ?,
                    percentage_error = ?,
                    squared_error = ?
                WHERE id = ?
            """,
                (
                    actual_price,
                    actual_timestamp,
                    absolute_error,
                    percentage_error,
                    squared_error,
                    forecast_id,
                ),
            )
            conn.commit()

        logger.info(
            f"Forecast {forecast_id} updated: actual=${actual_price:.2f}, "
            f"error={percentage_error:.2f}%"
        )
        return True

    def update_actuals_batch(
        self, symbol: str, current_price: float, max_age_hours: int = 48
    ) -> int:
        """
        Batch update all pending forecasts for a symbol.

        Args:
            symbol: Stock ticker
            current_price: Latest price
            max_age_hours: Only update forecasts within this window

        Returns:
            Number of forecasts updated
        """
        cutoff_time = time.time() - (max_age_hours * 3600)

        with sqlite3.connect(self.db_path) as conn:
            # Find pending forecasts
            rows = conn.execute(
                """
                SELECT id, forecast_price FROM forecasts
                WHERE symbol = ?
                  AND actual_price IS NULL
                  AND timestamp >= ?
            """,
                (symbol, cutoff_time),
            ).fetchall()

            if not rows:
                return 0

            # Update each forecast
            updated = 0
            for forecast_id, _forecast_price in rows:
                if self.update_actual(forecast_id, current_price):
                    updated += 1

        logger.info(f"Batch updated {updated} forecasts for {symbol} @ ${current_price:.2f}")
        return updated

    def calculate_metrics(self, symbol: str | None = None, days: int = 30) -> dict[str, Any]:
        """
        Calculate accuracy metrics.

        Args:
            symbol: Filter by symbol (None = all symbols)
            days: Look back window

        Returns:
            Dict with MAP, RMSE, bias, count
        """
        cutoff_time = time.time() - (days * 86400)

        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                rows = conn.execute(
                    """
                    SELECT
                        percentage_error,
                        squared_error,
                        forecast_price,
                        actual_price,
                        confidence
                    FROM forecasts
                    WHERE symbol = ?
                      AND actual_price IS NOT NULL
                      AND timestamp >= ?
                """,
                    (symbol, cutoff_time),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        percentage_error,
                        squared_error,
                        forecast_price,
                        actual_price,
                        confidence
                    FROM forecasts
                    WHERE actual_price IS NOT NULL
                      AND timestamp >= ?
                """,
                    (cutoff_time,),
                ).fetchall()

        if not rows:
            return {
                "error": "No completed forecasts found",
                "count": 0,
                "symbol": symbol,
                "days": days,
            }

        # Calculate metrics
        count = len(rows)
        percentage_errors = [r[0] for r in rows]
        squared_errors = [r[1] for r in rows]
        forecast_prices = [r[2] for r in rows]
        actual_prices = [r[3] for r in rows]
        confidences = [r[4] for r in rows if r[4] is not None]

        map = sum(percentage_errors) / count
        rmse = (sum(squared_errors) / count) ** 0.5

        # Bias: avg(forecast - actual)
        bias = sum(f - a for f, a in zip(forecast_prices, actual_prices, strict=False)) / count
        bias_pct = (bias / sum(actual_prices) * count * 100.0) if actual_prices else 0.0

        # Confidence statistics
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        return {
            "map": round(map, 4),
            "rmse": round(rmse, 4),
            "bias": round(bias, 4),
            "bias_pct": round(bias_pct, 4),
            "count": count,
            "avg_confidence": round(avg_confidence, 4) if avg_confidence else None,
            "symbol": symbol or "all",
            "days": days,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_accuracy_report(self, symbol: str | None = None, days: int = 30) -> dict[str, Any]:
        """
        Generate comprehensive accuracy report.

        Args:
            symbol: Filter by symbol (None = all)
            days: Look back window

        Returns:
            Dict with metrics, trends, and recommendations
        """
        metrics = self.calculate_metrics(symbol=symbol, days=days)

        if "error" in metrics:
            return metrics

        # Determine accuracy rating
        map = metrics["map"]
        if map < 2.0:
            rating = "excellent"
            color = "green"
        elif map < 5.0:
            rating = "good"
            color = "green"
        elif map < 10.0:
            rating = "fair"
            color = "yellow"
        else:
            rating = "poor"
            color = "red"

        # Recommendations
        recommendations = []
        if map > 5.0:
            recommendations.append("Consider model retuning (MAP > 5%)")
        if abs(metrics["bias_pct"]) > 3.0:
            direction = "over-predicting" if metrics["bias_pct"] > 0 else "under-predicting"
            recommendations.append(f"Model is {direction} by {abs(metrics['bias_pct']):.2f}%")
        if metrics["count"] < 10:
            recommendations.append("Limited sample size, continue collecting data")

        return {
            "metrics": metrics,
            "rating": rating,
            "rating_color": color,
            "recommendations": recommendations,
            "summary": (
                f"MAP: {map:.2f}% ({rating}), "
                f"RMSE: ${metrics['rmse']:.2f}, "
                f"Bias: {metrics['bias_pct']:+.2f}%, "
                f"n={metrics['count']}"
            ),
        }

    def get_recent_forecasts(
        self, symbol: str | None = None, limit: int = 10, include_pending: bool = True
    ) -> list[dict[str, Any]]:
        """
        Get recent forecasts with details.

        Args:
            symbol: Filter by symbol
            limit: Max forecasts to return
            include_pending: Include forecasts without actual prices

        Returns:
            List of forecast dicts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if symbol:
                if include_pending:
                    rows = conn.execute(
                        """
                        SELECT * FROM forecasts
                        WHERE symbol = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """,
                        (symbol, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM forecasts
                        WHERE symbol = ?
                          AND actual_price IS NOT NULL
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """,
                        (symbol, limit),
                    ).fetchall()
            else:
                if include_pending:
                    rows = conn.execute(
                        """
                        SELECT * FROM forecasts
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """,
                        (limit,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM forecasts
                        WHERE actual_price IS NOT NULL
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """,
                        (limit,),
                    ).fetchall()

        forecasts = []
        for row in rows:
            forecast = dict(row)
            # Parse metadata
            if forecast.get("metadata"):
                try:
                    forecast["metadata"] = json.loads(forecast["metadata"])
                except Exception:
                    pass
            forecasts.append(forecast)

        return forecasts

    def cleanup_old_forecasts(self, days: int = 90) -> int:
        """
        Delete forecasts older than N days.

        Args:
            days: Age threshold

        Returns:
            Number of forecasts deleted
        """
        cutoff_time = time.time() - (days * 86400)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM forecasts WHERE timestamp < ?", (cutoff_time,))
            deleted = cursor.rowcount
            conn.commit()

        logger.info(f"Cleaned up {deleted} forecasts older than {days} days")
        return deleted


# Singleton instance
_tracker = None


def get_accuracy_tracker() -> AccuracyTracker:
    """Get or create the global accuracy tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = AccuracyTracker()
    return _tracker


# Convenience functions
def record_forecast(*args, **kwargs) -> int:
    """Record a forecast (convenience wrapper)."""
    return get_accuracy_tracker().record_forecast(*args, **kwargs)


def update_actual(*args, **kwargs) -> bool:
    """Update forecast with actual price (convenience wrapper)."""
    return get_accuracy_tracker().update_actual(*args, **kwargs)


def get_accuracy_report(*args, **kwargs) -> dict[str, Any]:
    """Get accuracy report (convenience wrapper)."""
    return get_accuracy_tracker().get_accuracy_report(*args, **kwargs)


def calculate_metrics(*args, **kwargs) -> dict[str, Any]:
    """Calculate accuracy metrics (convenience wrapper)."""
    return get_accuracy_tracker().calculate_metrics(*args, **kwargs)


def calculate_accuracy(*args, **kwargs) -> dict[str, Any]:
    """Calculate accuracy (alias for calculate_metrics)."""
    return calculate_metrics(*args, **kwargs)
