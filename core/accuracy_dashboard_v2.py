#!/usr/bin/env python3
"""
GHOST Accuracy Dashboard v2 - PostgreSQL Edition
=================================================
Reads from ghost_prediction_outcomes table in production PostgreSQL database.

Author: Ghost Surgeon Omega
Date: 2025-12-04
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

LOGGER = logging.getLogger("ghost.accuracy_dashboard_v2")


class AccuracyDashboardV2:
    """PostgreSQL-based accuracy dashboard reading from ghost_prediction_outcomes."""
    
    def __init__(self):
        """Initialize dashboard with PostgreSQL connection."""
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            LOGGER.warning("DATABASE_URL not set")
        elif not HAS_PSYCOPG2:
            LOGGER.warning("psycopg2 not installed")
    
    def _get_connection(self):
        """Get PostgreSQL connection."""
        if not self.database_url or not HAS_PSYCOPG2:
            return None
        return psycopg2.connect(self.database_url)
    
    def get_dashboard_summary(self, days: int = 30) -> dict[str, Any]:
        """
        Get comprehensive dashboard summary from PostgreSQL.
        
        Reads from ghost_prediction_outcomes table where:
        - hit_direction = 1 means correct prediction
        - hit_direction = 0 means incorrect prediction
        
        Args:
            days: Lookback period (default 30 days)
        
        Returns:
            Dashboard metrics with accuracy, trends, symbols, etc.
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
            "incorrect": 0,
            "accuracy_trend": {"7d": None, "30d": None, "90d": None},
            "by_symbol": {},
            "by_confidence_band": {
                "40-60%": {"total": 0, "reconciled": 0, "correct": 0, "accuracy": 0.0, "avg_confidence": 0.0},
                "60-70%": {"total": 0, "reconciled": 0, "correct": 0, "accuracy": 0.0, "avg_confidence": 0.0},
                "70-85%": {"total": 0, "reconciled": 0, "correct": 0, "accuracy": 0.0, "avg_confidence": 0.0}
            },
            "calibration": {
                "avg_claimed_confidence": 0.0,
                "actual_accuracy": 0.0,
                "calibration_error": 0.0,
                "is_overconfident": False,
                "interpretation": ""
            },
            "recent_predictions": []
        }
        
        conn = self._get_connection()
        if not conn:
            LOGGER.warning("No database connection, returning empty summary")
            return summary
        
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                    # Total reconciled predictions
                    cursor.execute("""
                        SELECT COUNT(*) as count FROM ghost_prediction_outcomes
                        WHERE closed_at >= %s
                    """, (cutoff_dt,))
                    result = cursor.fetchone()
                    summary["reconciled"] = result["count"] if result else 0
                    summary["total_predictions"] = summary["reconciled"]
                    
                    # Correct predictions (hit_direction = 1)
                    cursor.execute("""
                        SELECT COUNT(*) as count FROM ghost_prediction_outcomes
                        WHERE closed_at >= %s AND hit_direction = 1
                    """, (cutoff_dt,))
                    result = cursor.fetchone()
                    summary["correct"] = result["count"] if result else 0
                    
                    # Incorrect predictions (hit_direction = 0)
                    cursor.execute("""
                        SELECT COUNT(*) as count FROM ghost_prediction_outcomes
                        WHERE closed_at >= %s AND hit_direction = 0
                    """, (cutoff_dt,))
                    result = cursor.fetchone()
                    summary["incorrect"] = result["count"] if result else 0
                    
                    # Calculate overall accuracy
                    if summary["reconciled"] > 0:
                        summary["overall_accuracy"] = round(
                            summary["correct"] / summary["reconciled"], 3
                        )
                    
                    # Accuracy trends (7d, 30d, 90d)
                    for period_name, period_days in [("7d", 7), ("30d", 30), ("90d", 90)]:
                        period_cutoff = datetime.now() - timedelta(days=period_days)
                        cursor.execute("""
                            SELECT 
                                COUNT(*) as total,
                                SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct
                            FROM ghost_prediction_outcomes
                            WHERE closed_at >= %s
                        """, (period_cutoff,))
                        result = cursor.fetchone()
                        if result and result["total"] > 0:
                            summary["accuracy_trend"][period_name] = round(
                                result["correct"] / result["total"], 3
                            )
                    
                    # By-symbol breakdown - DISABLED (symbol not in ghost_prediction_outcomes table)
                    # TODO: Add symbol column to ghost_prediction_outcomes or join with predictions table
                    # For now, by_symbol will remain empty
                    summary["by_symbol"] = {}
                    
                    # By-confidence band (using predicted_confidence)
                    cursor.execute("""
                        SELECT 
                            predicted_confidence,
                            hit_direction
                        FROM ghost_prediction_outcomes
                        WHERE closed_at >= %s
                        AND predicted_confidence IS NOT NULL
                    """, (cutoff_dt,))
                    
                    band_40_60 = []
                    band_60_70 = []
                    band_70_85 = []
                    
                    for row in cursor.fetchall():
                        conf = row["predicted_confidence"]
                        hit = row["hit_direction"]
                        
                        if 0.40 <= conf < 0.60:
                            band_40_60.append((conf, hit))
                        elif 0.60 <= conf < 0.70:
                            band_60_70.append((conf, hit))
                        elif 0.70 <= conf <= 0.85:
                            band_70_85.append((conf, hit))
                    
                    # Calculate band stats
                    for band_name, band_data in [
                        ("40-60%", band_40_60),
                        ("60-70%", band_60_70),
                        ("70-85%", band_70_85)
                    ]:
                        if band_data:
                            total = len(band_data)
                            correct = sum(1 for _, hit in band_data if hit == 1)
                            avg_conf = sum(conf for conf, _ in band_data) / total
                            accuracy = correct / total if total > 0 else 0.0
                            
                            summary["by_confidence_band"][band_name] = {
                                "total": total,
                                "reconciled": total,
                                "correct": correct,
                                "accuracy": round(accuracy, 3),
                                "avg_confidence": round(avg_conf, 3)
                            }
                    
                    # Calibration analysis
                    cursor.execute("""
                        SELECT 
                            AVG(predicted_confidence) as avg_confidence,
                            AVG(CASE WHEN hit_direction = 1 THEN 1.0 ELSE 0.0 END) as actual_accuracy
                        FROM ghost_prediction_outcomes
                        WHERE closed_at >= %s
                        AND predicted_confidence IS NOT NULL
                    """, (cutoff_dt,))
                    result = cursor.fetchone()
                    if result and result["avg_confidence"] is not None:
                        avg_conf = float(result["avg_confidence"])
                        actual_acc = float(result["actual_accuracy"])
                        cal_error = avg_conf - actual_acc
                        
                        summary["calibration"] = {
                            "avg_claimed_confidence": round(avg_conf, 3),
                            "actual_accuracy": round(actual_acc, 3),
                            "calibration_error": round(cal_error, 3),
                            "is_overconfident": cal_error > 0.05,
                            "interpretation": self._interpret_calibration(cal_error)
                        }
                    
                    # Recent predictions
                    cursor.execute("""
                        SELECT 
                            prediction_id,
                            closed_at,
                            price_at_prediction,
                            price_at_resolution,
                            predicted_direction,
                            actual_direction,
                            hit_direction,
                            predicted_confidence
                        FROM ghost_prediction_outcomes
                        WHERE closed_at >= %s
                        ORDER BY closed_at DESC
                        LIMIT 20
                    """, (cutoff_dt,))
                    
                    for row in cursor.fetchall():
                        summary["recent_predictions"].append({
                            "prediction_id": row["prediction_id"],
                            "closed_at": row["closed_at"].timestamp() if row["closed_at"] else None,
                            "predicted_price": row["price_at_prediction"],
                            "actual_price": row["price_at_resolution"],
                            "predicted_direction": row["predicted_direction"],
                            "actual_direction": row["actual_direction"],
                            "correct": row["hit_direction"] == 1,
                            "confidence": row["predicted_confidence"]
                        })
        
        except Exception as e:
            LOGGER.error(f"Dashboard query failed: {e}", exc_info=True)
        finally:
            conn.close()
        
        return summary
    
    def _interpret_calibration(self, error: float) -> str:
        """Interpret calibration error."""
        if abs(error) < 0.03:
            return "Well-calibrated predictions"
        elif error > 0.10:
            return f"Significantly overconfident by {error*100:.1f}%"
        elif error > 0.05:
            return f"Moderately overconfident by {error*100:.1f}%"
        elif error < -0.10:
            return f"Significantly underconfident by {abs(error)*100:.1f}%"
        elif error < -0.05:
            return f"Moderately underconfident by {abs(error)*100:.1f}%"
        else:
            return f"Slightly {'over' if error > 0 else 'under'}confident by {abs(error)*100:.1f}%"
    
    def get_performance_metrics(self, days: int = 30) -> dict[str, Any]:
        """
        Get advanced performance metrics (Sharpe ratio, drawdown, etc.).
        
        Note: Full implementation requires historical price tracking.
        For now, returns basic win/loss metrics.
        """
        conn = self._get_connection()
        if not conn:
            return {
                "win_rate": 0.0,
                "total_trades": 0,
                "best_symbol": None,
                "worst_symbol": None
            }
        
        cutoff_dt = datetime.now() - timedelta(days=days)
        
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                    # Overall win rate
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total,
                            COALESCE(SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END), 0) as wins
                        FROM ghost_prediction_outcomes
                        WHERE closed_at >= %s
                    """, (cutoff_dt,))
                    result = cursor.fetchone()
                    total = result["total"] if result else 0
                    wins = result["wins"] if result and result["wins"] is not None else 0
                    win_rate = wins / total if total > 0 else 0.0
                    
                    # Best/worst symbols - DISABLED (symbol not in ghost_prediction_outcomes table)
                    # TODO: Add symbol column or join with predictions table
                    best_symbol = None
                    worst_symbol = None
                    
                    return {
                        "win_rate": round(win_rate, 3),
                        "total_trades": total,
                        "wins": wins,
                        "losses": total - wins,
                        "best_symbol": best_symbol,
                        "worst_symbol": worst_symbol,
                        "sharpe_ratio": None,  # Not yet implemented
                        "max_drawdown_pct": None  # Not yet implemented
                    }
        except Exception as e:
            LOGGER.error(f"Performance metrics failed: {e}", exc_info=True)
            return {
                "win_rate": 0.0,
                "total_trades": 0,
                "error": str(e)
            }
        finally:
            conn.close()


# Singleton instance
_dashboard_instance = None

def get_accuracy_dashboard_v2():
    """Get singleton dashboard instance."""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = AccuracyDashboardV2()
    return _dashboard_instance
