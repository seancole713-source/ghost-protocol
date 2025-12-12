"""
📊 GHOST PERFORMANCE DASHBOARD
Real-time performance metrics, P&L tracking, win rates, accuracy over time
Lightweight web dashboard showing if Ghost is making money
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Database paths
FEEDBACK_DB = Path(__file__).parent.parent / "data" / "feedback_loop.db"
FORECASTS_DB = Path(__file__).parent.parent / "data" / "forecasts.db"

# Cache for dashboard metrics (refresh every 5 minutes)
_DASHBOARD_CACHE = {}
_CACHE_TIMESTAMP = 0
_CACHE_TTL = 300  # 5 minutes


def get_dashboard_metrics() -> dict[str, Any]:
    """
    Get comprehensive performance metrics for dashboard
    Returns: {
        "overall": {...},
        "today": {...},
        "last_7d": {...},
        "last_30d": {...},
        "by_asset_type": {...},
        "recent_predictions": [...]
    }
    """
    global _DASHBOARD_CACHE, _CACHE_TIMESTAMP
    
    # Return cached data if still fresh
    if time.time() - _CACHE_TIMESTAMP < _CACHE_TTL and _DASHBOARD_CACHE:
        return _DASHBOARD_CACHE
    
    try:
        metrics = {
            "overall": _get_overall_stats(),
            "today": _get_period_stats(hours=24),
            "last_7d": _get_period_stats(days=7),
            "last_30d": _get_period_stats(days=30),
            "by_asset_type": _get_stats_by_asset_type(),
            "recent_predictions": _get_recent_predictions(limit=20),
            "top_performers": _get_top_performers(limit=10),
            "worst_performers": _get_worst_performers(limit=10),
            "confidence_calibration": _get_confidence_calibration(),
            "generated_at": datetime.now().isoformat()
        }
        
        # Update cache
        _DASHBOARD_CACHE = metrics
        _CACHE_TIMESTAMP = time.time()
        
        return metrics
    
    except Exception as e:
        LOGGER.error(f"Failed to generate dashboard metrics: {e}", exc_info=True)
        return {
            "error": str(e),
            "overall": {"predictions": 0, "win_rate": 0},
            "generated_at": datetime.now().isoformat()
        }


def _get_overall_stats() -> dict[str, Any]:
    """Get all-time statistics"""
    if not FEEDBACK_DB.exists():
        return {"predictions": 0, "win_rate": 0, "avg_gain": 0}
    
    try:
        conn = sqlite3.connect(str(FEEDBACK_DB))
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM prediction_outcomes")
        total = cursor.fetchone()[0]
        
        # Win rate
        cursor.execute("SELECT COUNT(*) FROM prediction_outcomes WHERE was_correct = 1")
        wins = cursor.fetchone()[0]
        
        # Average accuracy %
        cursor.execute("SELECT AVG(accuracy_pct) FROM prediction_outcomes WHERE accuracy_pct IS NOT NULL")
        avg_accuracy = cursor.fetchone()[0] or 0
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM prediction_outcomes")
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Average gain (predicted vs actual)
        cursor.execute("""
            SELECT AVG((actual_price - predicted_price) / predicted_price * 100) 
            FROM prediction_outcomes 
            WHERE predicted_price > 0 AND actual_price > 0
        """)
        avg_gain = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "predictions": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round((wins / total * 100) if total > 0 else 0, 2),
            "avg_accuracy": round(avg_accuracy, 2),
            "avg_confidence": round(avg_confidence, 2),
            "avg_gain_pct": round(avg_gain, 2)
        }
    
    except Exception as e:
        LOGGER.error(f"Error getting overall stats: {e}")
        return {"predictions": 0, "win_rate": 0}


def _get_period_stats(hours: int = None, days: int = None) -> dict[str, Any]:
    """Get statistics for specific time period"""
    if not FEEDBACK_DB.exists():
        return {"predictions": 0, "win_rate": 0}
    
    try:
        # Calculate timestamp cutoff
        if hours:
            cutoff = time.time() - (hours * 3600)
        elif days:
            cutoff = time.time() - (days * 24 * 3600)
        else:
            return {}
        
        conn = sqlite3.connect(str(FEEDBACK_DB))
        cursor = conn.cursor()
        
        # Predictions in period
        cursor.execute(
            "SELECT COUNT(*) FROM prediction_outcomes WHERE timestamp >= ?",
            (cutoff,)
        )
        total = cursor.fetchone()[0]
        
        # Wins in period
        cursor.execute(
            "SELECT COUNT(*) FROM prediction_outcomes WHERE timestamp >= ? AND was_correct = 1",
            (cutoff,)
        )
        wins = cursor.fetchone()[0]
        
        # Average gain in period
        cursor.execute("""
            SELECT AVG((actual_price - predicted_price) / predicted_price * 100)
            FROM prediction_outcomes
            WHERE timestamp >= ? AND predicted_price > 0 AND actual_price > 0
        """, (cutoff,))
        avg_gain = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "predictions": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round((wins / total * 100) if total > 0 else 0, 2),
            "avg_gain_pct": round(avg_gain, 2)
        }
    
    except Exception as e:
        LOGGER.error(f"Error getting period stats: {e}")
        return {"predictions": 0, "win_rate": 0}


def _get_stats_by_asset_type() -> dict[str, Any]:
    """Get statistics broken down by asset type (stock vs crypto)"""
    # Note: prediction_outcomes table doesn't have asset_type column yet
    # This would require adding it or joining with predictions table
    # For now, return placeholder
    return {
        "stocks": {"predictions": 0, "win_rate": 0},
        "crypto": {"predictions": 0, "win_rate": 0}
    }


def _get_recent_predictions(limit: int = 20) -> list[dict[str, Any]]:
    """Get most recent predictions with outcomes"""
    if not FEEDBACK_DB.exists():
        return []
    
    try:
        conn = sqlite3.connect(str(FEEDBACK_DB))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                symbol,
                direction,
                confidence,
                predicted_price,
                actual_price,
                was_correct,
                accuracy_pct,
                timestamp
            FROM prediction_outcomes
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            predictions.append({
                "symbol": row[0],
                "direction": row[1],
                "confidence": round(row[2], 2),
                "predicted_price": round(row[3], 2) if row[3] else None,
                "actual_price": round(row[4], 2) if row[4] else None,
                "was_correct": bool(row[5]),
                "accuracy_pct": round(row[6], 2) if row[6] else None,
                "timestamp": datetime.fromtimestamp(row[7]).isoformat()
            })
        
        return predictions
    
    except Exception as e:
        LOGGER.error(f"Error getting recent predictions: {e}")
        return []


def _get_top_performers(limit: int = 10) -> list[dict[str, Any]]:
    """Get symbols with best win rates (min 5 predictions)"""
    if not FEEDBACK_DB.exists():
        return []
    
    try:
        conn = sqlite3.connect(str(FEEDBACK_DB))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                symbol,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                AVG(confidence) as avg_confidence
            FROM prediction_outcomes
            GROUP BY symbol
            HAVING total >= 5
            ORDER BY (CAST(wins AS FLOAT) / total) DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        performers = []
        for row in rows:
            win_rate = (row[2] / row[1] * 100) if row[1] > 0 else 0
            performers.append({
                "symbol": row[0],
                "predictions": row[1],
                "wins": row[2],
                "win_rate": round(win_rate, 2),
                "avg_confidence": round(row[3], 2)
            })
        
        return performers
    
    except Exception as e:
        LOGGER.error(f"Error getting top performers: {e}")
        return []


def _get_worst_performers(limit: int = 10) -> list[dict[str, Any]]:
    """Get symbols with worst win rates (min 5 predictions)"""
    if not FEEDBACK_DB.exists():
        return []
    
    try:
        conn = sqlite3.connect(str(FEEDBACK_DB))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                symbol,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                AVG(confidence) as avg_confidence
            FROM prediction_outcomes
            GROUP BY symbol
            HAVING total >= 5
            ORDER BY (CAST(wins AS FLOAT) / total) ASC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        performers = []
        for row in rows:
            win_rate = (row[2] / row[1] * 100) if row[1] > 0 else 0
            performers.append({
                "symbol": row[0],
                "predictions": row[1],
                "wins": row[2],
                "win_rate": round(win_rate, 2),
                "avg_confidence": round(row[3], 2)
            })
        
        return performers
    
    except Exception as e:
        LOGGER.error(f"Error getting worst performers: {e}")
        return []


def _get_confidence_calibration() -> dict[str, Any]:
    """
    Check if Ghost's confidence matches actual accuracy
    E.g., 80% confidence predictions should be correct 80% of the time
    """
    if not FEEDBACK_DB.exists():
        return {}
    
    try:
        conn = sqlite3.connect(str(FEEDBACK_DB))
        cursor = conn.cursor()
        
        # Group predictions by confidence ranges
        calibration = {}
        ranges = [
            (60, 70, "60-70%"),
            (70, 80, "70-80%"),
            (80, 90, "80-90%"),
            (90, 100, "90-100%")
        ]
        
        for min_conf, max_conf, label in ranges:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                    AVG(confidence) as avg_confidence
                FROM prediction_outcomes
                WHERE confidence >= ? AND confidence < ?
            """, (min_conf, max_conf))
            
            row = cursor.fetchone()
            if row and row[0] > 0:
                actual_accuracy = (row[1] / row[0] * 100) if row[0] > 0 else 0
                calibration[label] = {
                    "predictions": row[0],
                    "avg_confidence": round(row[2], 2),
                    "actual_accuracy": round(actual_accuracy, 2),
                    "calibration_error": round(abs(row[2] - actual_accuracy), 2)
                }
        
        conn.close()
        return calibration
    
    except Exception as e:
        LOGGER.error(f"Error calculating confidence calibration: {e}")
        return {}


def format_dashboard_text() -> str:
    """Format dashboard metrics for Telegram/terminal display"""
    metrics = get_dashboard_metrics()
    
    if "error" in metrics:
        return f"❌ Dashboard error: {metrics['error']}"
    
    overall = metrics["overall"]
    today = metrics["today"]
    
    text = "📊 **GHOST PERFORMANCE DASHBOARD**\n\n"
    
    # Overall stats
    text += "**ALL-TIME PERFORMANCE**\n"
    text += f"├─ Predictions: {overall['predictions']:,}\n"
    text += f"├─ Win Rate: {overall['win_rate']}%\n"
    text += f"├─ Avg Accuracy: {overall.get('avg_accuracy', 0)}%\n"
    text += f"├─ Avg Confidence: {overall.get('avg_confidence', 0)}%\n"
    text += f"└─ Avg Gain: {overall.get('avg_gain_pct', 0):+.2f}%\n\n"
    
    # Today's stats
    text += "**TODAY (24H)**\n"
    text += f"├─ Predictions: {today['predictions']}\n"
    text += f"├─ Wins: {today['wins']} | Losses: {today['losses']}\n"
    text += f"└─ Win Rate: {today['win_rate']}%\n\n"
    
    # Top performers
    top = metrics.get("top_performers", [])[:5]
    if top:
        text += "**TOP 5 PERFORMERS**\n"
        for i, perf in enumerate(top, 1):
            prefix = "└─" if i == len(top) else "├─"
            text += f"{prefix} {perf['symbol']}: {perf['win_rate']}% ({perf['predictions']} pred)\n"
        text += "\n"
    
    # Confidence calibration
    cal = metrics.get("confidence_calibration", {})
    if cal:
        text += "**CONFIDENCE CALIBRATION**\n"
        for label, data in cal.items():
            if data['predictions'] >= 10:  # Only show ranges with enough data
                text += f"├─ {label}: {data['actual_accuracy']}% actual (error: {data['calibration_error']}%)\n"
        text += "\n"
    
    text += f"🔄 Updated: {metrics['generated_at']}\n"
    text += "⚡ Auto-refreshes every 5 minutes"
    
    return text


async def dashboard_monitoring_loop():
    """Background task that monitors performance and sends alerts"""
    LOGGER.info("📊 Performance dashboard monitoring started")
    
    while True:
        try:
            await asyncio.sleep(3600)  # Check every hour
            
            # Get current metrics
            metrics = get_dashboard_metrics()
            
            # Alert if win rate drops below 50%
            today_win_rate = metrics.get("today", {}).get("win_rate", 0)
            if today_win_rate < 50 and metrics.get("today", {}).get("predictions", 0) >= 10:
                LOGGER.warning(f"⚠️ Today's win rate is low: {today_win_rate}%")
                # Could send Telegram alert here
            
            # Alert if confidence calibration error is high
            cal = metrics.get("confidence_calibration", {})
            for label, data in cal.items():
                if data.get("calibration_error", 0) > 15:  # >15% error
                    LOGGER.warning(
                        f"⚠️ Confidence calibration error high for {label}: "
                        f"{data['calibration_error']}%"
                    )
        
        except Exception as e:
            LOGGER.error(f"Dashboard monitoring error: {e}", exc_info=True)
            await asyncio.sleep(300)  # Wait 5 min on error


# Export main functions
__all__ = [
    "get_dashboard_metrics",
    "format_dashboard_text",
    "dashboard_monitoring_loop"
]
