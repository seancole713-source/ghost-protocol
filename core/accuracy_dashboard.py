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
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

LOGGER = logging.getLogger("ghost.accuracy_dashboard")


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
        """Get PostgreSQL connection."""
        if not self.database_url or not HAS_PSYCOPG2:
            return None
        return psycopg2.connect(self.database_url)
    
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
                
                # Total reconciled predictions (those with outcomes)
                cursor.execute("""
                    SELECT COUNT(*) FROM ghost_prediction_outcomes
                    WHERE closed_at >= %s
                """, (cutoff_dt,))
                summary["reconciled"] = cursor.fetchone()[0]
                summary["total_predictions"] = summary["reconciled"]  # For now, only count reconciled
                
                # Correct predictions
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM prediction_outcomes
                    WHERE predicted_at >= ?
                    AND correct = 1
                """, (cutoff_ts,))
                summary["correct"] = cursor.fetchone()[0]
                
                # Incorrect predictions
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM prediction_outcomes
                    WHERE predicted_at >= ?
                    AND correct = 0
                """, (cutoff_ts,))
                summary["incorrect"] = cursor.fetchone()[0]
                
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
    
    def _get_accuracy_trends(self) -> Dict[str, Optional[float]]:
        """Get accuracy trends for different time periods."""
        trends = {
            "7d": None,
            "30d": None,
            "90d": None
        }
        
        try:
            with sqlite3.connect(str(ACCURACY_DB)) as conn:
                for period, days in [("7d", 7), ("30d", 30), ("90d", 90)]:
                    cutoff_ts = time.time() - (days * 86400)
                    
                    cursor = conn.execute("""
                        SELECT 
                            COUNT(*) as total,
                            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_count
                        FROM prediction_outcomes
                        WHERE predicted_at >= ?
                        AND actual_price IS NOT NULL
                    """, (cutoff_ts,))
                    
                    row = cursor.fetchone()
                    if row and row[0] > 0:
                        trends[period] = round(row[1] / row[0], 3)
        
        except Exception as e:
            LOGGER.error(f"Failed to get accuracy trends: {e}")
        
        return trends
    
    def _get_accuracy_by_symbol(self, days: int) -> Dict[str, Dict[str, Any]]:
        """Get accuracy breakdown by symbol."""
        cutoff_ts = time.time() - (days * 86400)
        by_symbol = {}
        
        try:
            with sqlite3.connect(str(ACCURACY_DB)) as conn:
                cursor = conn.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as total,
                        SUM(CASE WHEN actual_price IS NOT NULL THEN 1 ELSE 0 END) as reconciled,
                        SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
                        AVG(confidence) as avg_confidence
                    FROM prediction_outcomes
                    WHERE predicted_at >= ?
                    GROUP BY symbol
                    ORDER BY total DESC
                """, (cutoff_ts,))
                
                for row in cursor:
                    symbol, total, reconciled, correct, avg_conf = row
                    
                    accuracy = 0.0
                    if reconciled > 0:
                        accuracy = round(correct / reconciled, 3)
                    
                    by_symbol[symbol] = {
                        "total_predictions": total,
                        "reconciled": reconciled,
                        "pending": total - reconciled,
                        "correct": correct,
                        "incorrect": reconciled - correct,
                        "accuracy": accuracy,
                        "avg_confidence": round(avg_conf, 3) if avg_conf else 0.0
                    }
        
        except Exception as e:
            LOGGER.error(f"Failed to get accuracy by symbol: {e}")
        
        return by_symbol
    
    def _get_accuracy_by_confidence(self, days: int) -> Dict[str, Dict[str, Any]]:
        """Get accuracy breakdown by confidence bands."""
        cutoff_ts = time.time() - (days * 86400)
        
        bands = {
            "40-60%": {"min": 0.40, "max": 0.60},
            "60-70%": {"min": 0.60, "max": 0.70},
            "70-85%": {"min": 0.70, "max": 0.85}
        }
        
        results = {}
        
        try:
            with sqlite3.connect(str(ACCURACY_DB)) as conn:
                for band_name, band_range in bands.items():
                    cursor = conn.execute("""
                        SELECT 
                            COUNT(*) as total,
                            SUM(CASE WHEN actual_price IS NOT NULL THEN 1 ELSE 0 END) as reconciled,
                            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
                            AVG(confidence) as avg_confidence
                        FROM prediction_outcomes
                        WHERE predicted_at >= ?
                        AND confidence >= ?
                        AND confidence < ?
                    """, (cutoff_ts, band_range["min"], band_range["max"]))
                    
                    row = cursor.fetchone()
                    if row:
                        total, reconciled, correct, avg_conf = row
                        
                        accuracy = 0.0
                        if reconciled and reconciled > 0:
                            accuracy = round(correct / reconciled, 3)
                        
                        results[band_name] = {
                            "total": total,
                            "reconciled": reconciled or 0,
                            "correct": correct or 0,
                            "accuracy": accuracy,
                            "avg_confidence": round(avg_conf, 3) if avg_conf else 0.0
                        }
        
        except Exception as e:
            LOGGER.error(f"Failed to get accuracy by confidence: {e}")
        
        return results
    
    def _get_calibration_analysis(self, days: int) -> Dict[str, Any]:
        """
        Analyze confidence calibration.
        
        Calibration error = claimed confidence - actual accuracy
        Example: If predictions with 75% confidence have 68% accuracy,
        calibration error is +7% (overconfident).
        """
        cutoff_ts = time.time() - (days * 86400)
        
        calibration = {
            "avg_claimed_confidence": 0.0,
            "actual_accuracy": 0.0,
            "calibration_error": 0.0,
            "is_overconfident": False,
            "interpretation": ""
        }
        
        try:
            with sqlite3.connect(str(ACCURACY_DB)) as conn:
                cursor = conn.execute("""
                    SELECT 
                        AVG(confidence) as avg_conf,
                        COUNT(*) as total,
                        SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct
                    FROM prediction_outcomes
                    WHERE predicted_at >= ?
                    AND actual_price IS NOT NULL
                """, (cutoff_ts,))
                
                row = cursor.fetchone()
                if row and row[1] > 0:
                    avg_conf, total, correct = row
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
        
        except Exception as e:
            LOGGER.error(f"Failed to get calibration analysis: {e}")
        
        return calibration
    
    def _get_recent_predictions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent predictions with outcomes."""
        predictions = []
        
        try:
            with sqlite3.connect(str(ACCURACY_DB)) as conn:
                cursor = conn.execute("""
                    SELECT 
                        symbol,
                        predicted_at,
                        check_at,
                        predicted_direction,
                        actual_direction,
                        confidence,
                        predicted_price,
                        actual_price,
                        correct
                    FROM prediction_outcomes
                    ORDER BY predicted_at DESC
                    LIMIT ?
                """, (limit,))
                
                for row in cursor:
                    (symbol, pred_at, check_at, pred_dir, actual_dir, 
                     conf, pred_price, actual_price, correct) = row
                    
                    predictions.append({
                        "symbol": symbol,
                        "predicted_at": int(pred_at),
                        "check_at": int(check_at),
                        "predicted_direction": pred_dir,
                        "actual_direction": actual_dir,
                        "confidence": round(conf, 3),
                        "predicted_price": pred_price,
                        "actual_price": actual_price,
                        "correct": correct,
                        "outcome_status": self._get_outcome_status(
                            check_at, actual_price, correct
                        )
                    })
        
        except Exception as e:
            LOGGER.error(f"Failed to get recent predictions: {e}")
        
        return predictions
    
    def _get_outcome_status(
        self, check_at: float, actual_price: Optional[float], correct: Optional[int]
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
    
    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get advanced performance metrics.
        
        Includes:
        - Sharpe ratio (if we have returns data)
        - Max drawdown
        - Win rate by day of week
        - Prediction latency
        """
        cutoff_ts = time.time() - (days * 86400)
        
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
        
        try:
            with sqlite3.connect(str(ACCURACY_DB)) as conn:
                # Get basic stats
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        AVG(CASE 
                            WHEN correct = 1 THEN 1.0 
                            ELSE 0.0 
                        END) as win_rate,
                        AVG(CASE 
                            WHEN actual_price IS NOT NULL AND predicted_price > 0
                            THEN ((actual_price - predicted_price) / predicted_price) * 100
                            ELSE NULL
                        END) as avg_return
                    FROM prediction_outcomes
                    WHERE predicted_at >= ?
                    AND actual_price IS NOT NULL
                """, (cutoff_ts,))
                
                row = cursor.fetchone()
                if row:
                    metrics["total_predictions"] = row[0] or 0
                    metrics["win_rate"] = round(row[1], 3) if row[1] else 0.0
                    metrics["avg_return_pct"] = round(row[2], 2) if row[2] else 0.0
                
                # Find best and worst performing symbols
                cursor = conn.execute("""
                    SELECT 
                        symbol,
                        AVG(CASE WHEN correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
                        COUNT(*) as count
                    FROM prediction_outcomes
                    WHERE predicted_at >= ?
                    AND actual_price IS NOT NULL
                    GROUP BY symbol
                    HAVING count >= 5
                    ORDER BY accuracy DESC
                """, (cutoff_ts,))
                
                results = cursor.fetchall()
                if results:
                    best = results[0]
                    worst = results[-1]
                    
                    metrics["best_symbol"] = {
                        "symbol": best[0],
                        "accuracy": round(best[1], 3),
                        "count": best[2]
                    }
                    
                    metrics["worst_symbol"] = {
                        "symbol": worst[0],
                        "accuracy": round(worst[1], 3),
                        "count": worst[2]
                    }
        
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
