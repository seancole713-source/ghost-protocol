#!/usr/bin/env python3
"""
GHOST Accuracy Dashboard - Comprehensive Outcome Reconciliation
================================================================
Real-time accuracy tracking, prediction outcomes, and performance analytics.

This is the CORE DASHBOARD for measuring Ghost's path to 70% accuracy.

Features:
- Real-time accuracy metrics (overall, by symbol, by timeframe)
- Prediction outcome tracking (correct/incorrect/pending)
- Confidence calibration analysis (claimed vs actual)
- Historical accuracy trends (7d, 30d, 90d)
- Performance by confidence band (40-60%, 60-70%, 70-85%)
- Win rate with stop loss simulation
- Symbol-level breakdown

Author: Ghost Surgeon Omega
Date: 2025-01-15
"""

import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

LOGGER = logging.getLogger("ghost.accuracy_dashboard")


# Optional legacy SQLite backing for some helper views in this module.
# Defaults to the Wolf SQLite DB if present.
ACCURACY_DB = Path(os.getenv("ACCURACY_DB_PATH") or os.getenv("WOLF_SQLITE_PATH") or "data/wolf.db")


class AccuracyDashboard:
    """
    Comprehensive accuracy dashboard for Ghost predictions.

    Provides real-time insights into prediction performance,
    accuracy trends, and calibration quality.

    Reads from PostgreSQL ghost_prediction_outcomes table.
    """

    def __init__(self):
        """Initialize dashboard with database connection."""
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            LOGGER.warning("DATABASE_URL not set, dashboard will return empty data")
        elif not HAS_PSYCOPG2:
            LOGGER.warning("psycopg2 not installed, dashboard will return empty data")

    def _get_connection(self):
        """Get PostgreSQL connection via shared pool bridge."""
        if not self.database_url:
            return None
        try:
            from core.db_pool import get_sync_connection
            return get_sync_connection().__enter__()
        except Exception:
            return None

    def get_dashboard_summary(self, days: int = 30) -> dict[str, Any]:
        """
        Get comprehensive dashboard summary from PostgreSQL.

        Reads from ghost_prediction_outcomes table.

        Args:
            days: Lookback period (default 30 days)

        Returns:
            {
                "timestamp": 1736899200,
                "period_days": 30,
                "overall_accuracy": 0.68,
                "total_predictions": 150,
                "reconciled": 120,
                "pending": 30,
                "correct": 82,
                "incorrect": 38,
                "accuracy_trend": {...},
                "by_symbol": {...},
                "by_confidence_band": {...},
                "calibration": {...},
                "recent_predictions": [...]
            }
        """
        cutoff_dt = datetime.now() - timedelta(days=days)

        summary = {
            "timestamp": int(time.time()),
            "period_days": days,
            "overall_accuracy": 0.0,
            "total_predictions": 0,
            "reconciled": 0,
            "pending": 0,
            "correct": 0,
            "incorrect": 0
        }

        # Get overall stats from PostgreSQL
        conn = self._get_connection()
        if not conn:
            LOGGER.warning("No database connection, returning empty summary")
            summary.update({
                "accuracy_trend": {"7d": None, "30d": None, "90d": None},
                "by_symbol": {},
                "by_confidence_band": self._empty_confidence_bands(),
                "calibration": self._empty_calibration(),
                "recent_predictions": []
            })
            return summary

        try:
            with conn:
                cursor = conn.cursor()

                # Outcome summary from PostgreSQL outcomes table
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS total_outcomes,
                        SUM(CASE WHEN status = 'completed' AND hit_direction IS NOT NULL THEN 1 ELSE 0 END) AS evaluated,
                        SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) AS correct,
                        SUM(CASE WHEN hit_direction = 0 THEN 1 ELSE 0 END) AS incorrect
                    FROM ghost_prediction_outcomes
                    WHERE closed_at >= %s
                    """,
                    (cutoff_dt,),
                )
                row = cursor.fetchone() or (0, 0, 0, 0)
                total_outcomes, evaluated, correct, incorrect = [int(x or 0) for x in row]

                summary["total_predictions"] = total_outcomes
                summary["reconciled"] = evaluated
                summary["correct"] = correct
                summary["incorrect"] = incorrect
                summary["pending"] = max(0, total_outcomes - evaluated)

                # Calculate accuracy
                if summary["reconciled"] > 0:
                    summary["overall_accuracy"] = round(
                        summary["correct"] / summary["reconciled"], 3
                    )

        except Exception as e:
            LOGGER.error(f"Failed to get overall stats: {e}")

        # Add accuracy trends
        summary["accuracy_trend"] = self._get_accuracy_trends()

        # Add by-symbol breakdown
        summary["by_symbol"] = self._get_accuracy_by_symbol(days)

        # Add confidence band analysis
        summary["by_confidence_band"] = self._get_accuracy_by_confidence(days)

        # Add calibration analysis
        summary["calibration"] = self._get_calibration_analysis(days)

        # Add recent predictions
        summary["recent_predictions"] = self._get_recent_predictions(limit=20)

        return summary

    def _empty_confidence_bands(self):
        return {}

    def _empty_calibration(self):
        return {
            "avg_claimed_confidence": 0.0,
            "actual_accuracy": 0.0,
            "calibration_error": 0.0,
            "is_overconfident": False,
            "interpretation": ""
        }

    def _get_accuracy_trends(self) -> dict[str, float | None]:
        """Get accuracy trends for different time periods from PostgreSQL."""
        trends = {
            "7d": None,
            "30d": None,
            "90d": None
        }

        conn = self._get_connection()
        if not conn:
            return trends

        try:
            cursor = conn.cursor()
            for period, days in [("7d", 7), ("30d", 30), ("90d", 90)]:
                cutoff_dt = datetime.now() - timedelta(days=days)
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct_count
                    FROM ghost_prediction_outcomes
                    WHERE closed_at >= %s
                    AND hit_direction IS NOT NULL
                """, (cutoff_dt,))

                row = cursor.fetchone()
                if row and row[0] and row[0] > 0:
                    trends[period] = round((row[1] or 0) / row[0], 3)
            cursor.close()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to get accuracy trends: {e}")

        return trends

    def _get_accuracy_by_symbol(self, days: int) -> dict[str, dict[str, Any]]:
        """Get accuracy breakdown by symbol from PostgreSQL."""
        by_symbol = {}
        conn = self._get_connection()
        if not conn:
            return by_symbol

        try:
            cutoff_dt = datetime.now() - timedelta(days=days)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    symbol,
                    COUNT(*) as total,
                    SUM(CASE WHEN hit_direction IS NOT NULL THEN 1 ELSE 0 END) as reconciled,
                    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(predicted_confidence) as avg_confidence
                FROM ghost_prediction_outcomes
                WHERE closed_at >= %s
                GROUP BY symbol
                ORDER BY total DESC
            """, (cutoff_dt,))

            for row in cursor.fetchall():
                symbol, total, reconciled, correct, avg_conf = row
                reconciled = int(reconciled or 0)
                correct = int(correct or 0)

                accuracy = 0.0
                if reconciled > 0:
                    accuracy = round(correct / reconciled, 3)

                by_symbol[symbol] = {
                    "total_predictions": int(total or 0),
                    "reconciled": reconciled,
                    "pending": int(total or 0) - reconciled,
                    "correct": correct,
                    "incorrect": reconciled - correct,
                    "accuracy": accuracy,
                    "avg_confidence": round(float(avg_conf or 0), 3)
                }
            cursor.close()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to get accuracy by symbol: {e}")

        return by_symbol

    def _get_accuracy_by_confidence(self, days: int) -> dict[str, dict[str, Any]]:
        """Get accuracy breakdown by confidence bands from PostgreSQL."""
        bands = {
            "40-60%": {"min": 0.40, "max": 0.60},
            "60-70%": {"min": 0.60, "max": 0.70},
            "70-85%": {"min": 0.70, "max": 0.85}
        }

        results = {}
        conn = self._get_connection()
        if not conn:
            return results

        try:
            cutoff_dt = datetime.now() - timedelta(days=days)
            cursor = conn.cursor()
            for band_name, band_range in bands.items():
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN hit_direction IS NOT NULL THEN 1 ELSE 0 END) as reconciled,
                        SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct,
                        AVG(predicted_confidence) as avg_confidence
                    FROM ghost_prediction_outcomes
                    WHERE closed_at >= %s
                    AND predicted_confidence >= %s
                    AND predicted_confidence < %s
                """, (cutoff_dt, band_range["min"], band_range["max"]))

                row = cursor.fetchone()
                if row:
                    total, reconciled, correct, avg_conf = row
                    reconciled = int(reconciled or 0)
                    correct = int(correct or 0)

                    accuracy = 0.0
                    if reconciled > 0:
                        accuracy = round(correct / reconciled, 3)

                    results[band_name] = {
                        "total": int(total or 0),
                        "reconciled": reconciled,
                        "correct": correct,
                        "accuracy": accuracy,
                        "avg_confidence": round(float(avg_conf or 0), 3)
                    }
            cursor.close()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to get accuracy by confidence: {e}")

        return results

    def _get_calibration_analysis(self, days: int) -> dict[str, Any]:
        """
        Analyze confidence calibration from PostgreSQL.

        Calibration error = claimed confidence - actual accuracy
        Example: If predictions with 75% confidence have 68% accuracy,
        calibration error is +7% (overconfident).
        """
        calibration = {
            "avg_claimed_confidence": 0.0,
            "actual_accuracy": 0.0,
            "calibration_error": 0.0,
            "is_overconfident": False,
            "interpretation": ""
        }

        conn = self._get_connection()
        if not conn:
            return calibration

        try:
            cutoff_dt = datetime.now() - timedelta(days=days)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    AVG(predicted_confidence) as avg_conf,
                    COUNT(*) as total,
                    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct
                FROM ghost_prediction_outcomes
                WHERE closed_at >= %s
                AND hit_direction IS NOT NULL
            """, (cutoff_dt,))

            row = cursor.fetchone()
            if row and row[1] and row[1] > 0:
                avg_conf = float(row[0] or 0)
                total = int(row[1])
                correct = int(row[2] or 0)
                actual_acc = correct / total

                calibration["avg_claimed_confidence"] = round(avg_conf, 3)
                calibration["actual_accuracy"] = round(actual_acc, 3)
                calibration["calibration_error"] = round(avg_conf - actual_acc, 3)
                calibration["is_overconfident"] = (avg_conf > actual_acc)

                # Interpretation
                error_pct = abs(calibration["calibration_error"] * 100)
                if error_pct < 3:
                    calibration["interpretation"] = "Well calibrated"
                elif error_pct < 5:
                    calibration["interpretation"] = "Slightly miscalibrated"
                elif calibration["is_overconfident"]:
                    calibration["interpretation"] = f"Overconfident by {error_pct:.1f}%"
                else:
                    calibration["interpretation"] = f"Underconfident by {error_pct:.1f}%"
            cursor.close()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to get calibration analysis: {e}")

        return calibration

    def _get_recent_predictions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent predictions with outcomes from PostgreSQL."""
        predictions = []
        conn = self._get_connection()
        if not conn:
            return predictions

        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    symbol,
                    EXTRACT(EPOCH FROM created_at) as predicted_at,
                    EXTRACT(EPOCH FROM closed_at) as check_at,
                    predicted_direction,
                    actual_direction,
                    predicted_confidence,
                    predicted_price,
                    close_price,
                    hit_direction
                FROM ghost_prediction_outcomes
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            for row in cursor.fetchall():
                (symbol, pred_at, check_at, pred_dir, actual_dir,
                 conf, pred_price, actual_price, correct) = row

                pred_at = float(pred_at or 0)
                check_at = float(check_at or 0)

                predictions.append({
                    "symbol": symbol,
                    "predicted_at": int(pred_at),
                    "check_at": int(check_at),
                    "predicted_direction": pred_dir,
                    "actual_direction": actual_dir,
                    "confidence": round(float(conf or 0), 3),
                    "predicted_price": float(pred_price) if pred_price else None,
                    "actual_price": float(actual_price) if actual_price else None,
                    "correct": int(correct) if correct is not None else None,
                    "outcome_status": self._get_outcome_status(
                        check_at, actual_price, correct
                    )
                })
            cursor.close()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to get recent predictions: {e}")

        return predictions

    def _get_outcome_status(
        self, check_at: float, actual_price: float | None, correct: int | None
    ) -> str:
        """Determine outcome status for a prediction."""
        current_time = time.time()

        # If check_at hasn't passed yet
        if check_at > current_time:
            remaining_hours = (check_at - current_time) / 3600
            return f"PENDING ({remaining_hours:.1f}h remaining)"

        # If check_at has passed but no outcome
        if actual_price is None:
            return "AWAITING_RECONCILIATION"

        # If outcome recorded
        if correct == 1:
            return "CORRECT ✅"
        elif correct == 0:
            return "INCORRECT ❌"
        else:
            return "UNKNOWN"

    def get_performance_metrics(self, days: int = 30) -> dict[str, Any]:
        """
        Get advanced performance metrics from PostgreSQL.
        """
        metrics = {
            "period_days": days,
            "total_predictions": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
            "best_symbol": None,
            "worst_symbol": None
        }

        conn = self._get_connection()
        if not conn:
            return metrics

        try:
            cutoff_dt = datetime.now() - timedelta(days=days)
            cursor = conn.cursor()

            # Get basic stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(CASE
                        WHEN hit_direction = 1 THEN 1.0
                        ELSE 0.0
                    END) as win_rate,
                    AVG(CASE
                        WHEN close_price IS NOT NULL AND predicted_price > 0
                        THEN ((close_price - predicted_price) / predicted_price) * 100
                        ELSE NULL
                    END) as avg_return
                FROM ghost_prediction_outcomes
                WHERE closed_at >= %s
                AND hit_direction IS NOT NULL
            """, (cutoff_dt,))

            row = cursor.fetchone()
            if row:
                metrics["total_predictions"] = int(row[0] or 0)
                metrics["win_rate"] = round(float(row[1] or 0), 3)
                metrics["avg_return_pct"] = round(float(row[2] or 0), 2)

            # Find best and worst performing symbols
            cursor.execute("""
                SELECT
                    symbol,
                    AVG(CASE WHEN hit_direction = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
                    COUNT(*) as count
                FROM ghost_prediction_outcomes
                WHERE closed_at >= %s
                AND hit_direction IS NOT NULL
                GROUP BY symbol
                HAVING COUNT(*) >= 5
                ORDER BY accuracy DESC
            """, (cutoff_dt,))

            results = cursor.fetchall()
            if results:
                best = results[0]
                worst = results[-1]

                metrics["best_symbol"] = {
                    "symbol": best[0],
                    "accuracy": round(float(best[1] or 0), 3),
                    "count": int(best[2])
                }

                metrics["worst_symbol"] = {
                    "symbol": worst[0],
                    "accuracy": round(float(worst[1] or 0), 3),
                    "count": int(worst[2])
                }
            cursor.close()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to get performance metrics: {e}")

        return metrics


# Singleton instance
_dashboard_instance = None


def get_accuracy_dashboard() -> AccuracyDashboard:
    """Get singleton accuracy dashboard instance."""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = AccuracyDashboard()
    return _dashboard_instance
