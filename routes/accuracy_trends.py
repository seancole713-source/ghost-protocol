"""
Accuracy Trending API (Phase 3.8)

Provides daily/weekly/monthly accuracy trends for charting.

Ghost Protocol v5 — Session 6
"""

from fastapi import APIRouter
import logging
from typing import Dict, List
import time

router = APIRouter()
LOGGER = logging.getLogger("ghost.accuracy")


@router.get("/api/accuracy/trends")
async def api_accuracy_trends(days: int = 30):
    """
    Get accuracy trends over time for charting.
    
    Args:
        days: Number of days to fetch (default 30)
        
    Returns:
        {
            "ok": bool,
            "daily": List[{"date": str, "accuracy": float, "total": int}],
            "weekly": List[{"week": str, "accuracy": float, "total": int}],
            "overall": {"accuracy": float, "total": int}
        }
    """
    try:
        from core.db_pool import get_sync_connection
        from datetime import datetime, timedelta
        
        cutoff_ts = int(time.time()) - (days * 24 * 3600)
        
        with get_sync_connection() as conn:
            cur = conn.cursor()
            
            # Daily accuracy trends
            cur.execute("""
                SELECT 
                    DATE(to_timestamp(predicted_at)) as date,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct
                FROM ghost_predictions
                WHERE correct IS NOT NULL
                  AND predicted_at > %s
                GROUP BY DATE(to_timestamp(predicted_at))
                ORDER BY date DESC
                LIMIT %s
            """, (cutoff_ts, days))
            
            daily_rows = cur.fetchall()
            daily_trends = []
            for date, total, correct in daily_rows:
                accuracy = (correct / total * 100) if total > 0 else 0
                daily_trends.append({
                    "date": str(date),
                    "accuracy": round(accuracy, 1),
                    "correct": correct or 0,
                    "total": total
                })
            
            # Weekly accuracy trends  
            cur.execute("""
                SELECT 
                    DATE_TRUNC('week', to_timestamp(predicted_at)) as week_start,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct
                FROM ghost_predictions
                WHERE correct IS NOT NULL
                  AND predicted_at > %s
                GROUP BY DATE_TRUNC('week', to_timestamp(predicted_at))
                ORDER BY week_start DESC
                LIMIT 12
            """, (cutoff_ts,))
            
            weekly_rows = cur.fetchall()
            weekly_trends = []
            for week_start, total, correct in weekly_rows:
                accuracy = (correct / total * 100) if total > 0 else 0
                weekly_trends.append({
                    "week": str(week_start.date()),
                    "accuracy": round(accuracy, 1),
                    "correct": correct or 0,
                    "total": total
                })
            
            # Overall accuracy
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct
                FROM ghost_predictions
                WHERE correct IS NOT NULL
                  AND predicted_at > %s
            """, (cutoff_ts,))
            
            total, correct = cur.fetchone()
            overall_accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            "ok": True,
            "daily": daily_trends,
            "weekly": weekly_trends,
            "overall": {
                "accuracy": round(overall_accuracy, 1),
                "correct": correct or 0,
                "total": total
            },
            "days": days
        }
        
    except Exception as e:
        LOGGER.error(f"Accuracy trends failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "daily": [],
            "weekly": [],
            "overall": {"accuracy": 0.0, "correct": 0, "total": 0}
        }
