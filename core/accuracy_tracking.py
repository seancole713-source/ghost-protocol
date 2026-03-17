#!/usr/bin/env python3
"""
Real-Time Accuracy Tracking & Analytics
========================================
Advanced tracking features for monitoring prediction performance:
- Historical accuracy trending
- Confidence score correlation analysis
- Performance degradation alerts
"""

import logging
import time
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import statistics

# NOTE: get_live_accuracy_dashboard import removed (Step 4C, Mar 17 2026).
# Snapshot now reads PostgreSQL directly instead of making live Coinbase HTTP calls.
# Functions that still need it import lazily below.

def _get_pg_accuracy() -> Dict[str, Any]:
    """Get accuracy from PostgreSQL ghost_predictions (deterministic, no HTTP)."""
    import os
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return {"ok": False, "current_accuracy_pct": 0, "total_predictions": 0,
                "correct_now": 0, "wrong_now": 0, "predictions": []}
    try:
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            try:
                conn.rollback()
            except Exception:
                pass
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END), 0) AS wins
                FROM ghost_predictions
                WHERE checked = 1
                  AND eval_version NOT LIKE 'skip%%'
            """)
            row = cur.fetchone()
            total = row[0] if row else 0
            correct = row[1] if row else 0
            cur.close()
        acc = round(correct / total * 100, 1) if total > 0 else 0.0
        return {"ok": True, "current_accuracy_pct": acc, "total_predictions": total,
                "correct_now": correct, "wrong_now": total - correct, "predictions": []}
    except Exception:
        return {"ok": False, "current_accuracy_pct": 0, "total_predictions": 0,
                "correct_now": 0, "wrong_now": 0, "predictions": []}

LOGGER = logging.getLogger("ghost.accuracy_tracking")

# In-memory cache for trending (resets on restart, will use Redis in production)
ACCURACY_HISTORY: List[Dict[str, Any]] = []
MAX_HISTORY_POINTS = 1000


def record_accuracy_snapshot():
    """
    Record current accuracy for trending analysis.
    Called periodically (e.g., every 5 minutes) by background job.

    FIX (Step 4C, Mar 17 2026): Was calling get_live_accuracy_dashboard() which
    made live Coinbase HTTP calls per prediction → any HTTP failure = skip → count
    oscillated wildly (5831→4→0→73→901). Also only tracked 20 hardcoded crypto
    symbols, missing all stock predictions.

    Now reads directly from PostgreSQL ghost_predictions (authoritative evaluator
    table). Count is deterministic — no HTTP calls, no Coinbase dependency.
    """
    try:
        import os
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return

        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            try:
                conn.rollback()
            except Exception:
                pass
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END), 0) AS wins
                FROM ghost_predictions
                WHERE checked = 1
                  AND eval_version NOT LIKE 'skip%%'
            """)
            row = cur.fetchone()
            total = row[0] if row else 0
            correct = row[1] if row else 0
            cur.close()

        if total == 0:
            return

        accuracy_pct = round(correct / total * 100, 1)
        wrong = total - correct

        snapshot = {
            "timestamp": time.time(),
            "accuracy_pct": accuracy_pct,
            "total_predictions": total,
            "correct_now": correct,
            "wrong_now": wrong,
        }

        ACCURACY_HISTORY.append(snapshot)

        # Keep only recent history
        if len(ACCURACY_HISTORY) > MAX_HISTORY_POINTS:
            ACCURACY_HISTORY.pop(0)

        LOGGER.info(f"Recorded accuracy snapshot: {accuracy_pct:.1f}% ({correct}/{total})")

    except Exception as e:
        LOGGER.error(f"Failed to record accuracy snapshot: {e}", exc_info=True)


def get_accuracy_trending(hours: int = 24) -> Dict[str, Any]:
    """
    Get accuracy trending over time.
    
    Args:
        hours: Lookback period (default 24 hours)
        
    Returns:
        {
            "ok": true,
            "period_hours": 24,
            "data_points": 288,
            "current_accuracy": 90.0,
            "avg_accuracy": 87.5,
            "min_accuracy": 75.0,
            "max_accuracy": 95.0,
            "trend": "improving",
            "history": [
                {"timestamp": 1234567890, "accuracy_pct": 85.0},
                ...
            ]
        }
    """
    try:
        if not ACCURACY_HISTORY:
            # No history yet, get current snapshot from PostgreSQL
            current = _get_pg_accuracy()
            return {
                "ok": True,
                "period_hours": hours,
                "data_points": 0,
                "current_accuracy": current.get("current_accuracy_pct", 0.0),
                "avg_accuracy": current.get("current_accuracy_pct", 0.0),
                "min_accuracy": current.get("current_accuracy_pct", 0.0),
                "max_accuracy": current.get("current_accuracy_pct", 0.0),
                "trend": "insufficient_data",
                "history": [],
                "message": "Accuracy history not yet available. Wait 5-10 minutes for data collection."
            }
        
        # Filter to requested time window
        cutoff_time = time.time() - (hours * 3600)
        recent_history = [
            h for h in ACCURACY_HISTORY
            if h["timestamp"] >= cutoff_time
        ]
        
        if not recent_history:
            recent_history = ACCURACY_HISTORY[-10:]  # At least show last 10 points
        
        # Calculate statistics
        accuracies = [h["accuracy_pct"] for h in recent_history]
        current_acc = accuracies[-1] if accuracies else 0.0
        avg_acc = statistics.mean(accuracies) if accuracies else 0.0
        min_acc = min(accuracies) if accuracies else 0.0
        max_acc = max(accuracies) if accuracies else 0.0
        
        # Determine trend (compare first half vs second half)
        if len(accuracies) >= 4:
            mid = len(accuracies) // 2
            first_half_avg = statistics.mean(accuracies[:mid])
            second_half_avg = statistics.mean(accuracies[mid:])
            
            if second_half_avg > first_half_avg + 2.0:
                trend = "improving"
            elif second_half_avg < first_half_avg - 2.0:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "ok": True,
            "period_hours": hours,
            "data_points": len(recent_history),
            "current_accuracy": current_acc,
            "avg_accuracy": avg_acc,
            "min_accuracy": min_acc,
            "max_accuracy": max_acc,
            "trend": trend,
            "history": [
                {"timestamp": h["timestamp"], "accuracy_pct": h["accuracy_pct"]}
                for h in recent_history
            ]
        }
        
    except Exception as e:
        LOGGER.error(f"Accuracy trending failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "period_hours": hours,
            "data_points": 0
        }


def get_confidence_correlation() -> Dict[str, Any]:
    """
    Analyze correlation between confidence scores and actual accuracy.
    
    Shows if high-confidence predictions are actually more accurate.
    
    Returns:
        {
            "ok": true,
            "confidence_buckets": {
                "60-70%": {"count": 10, "accuracy": 85.0},
                "70-80%": {"count": 20, "accuracy": 90.0},
                "80-90%": {"count": 5, "accuracy": 80.0}
            },
            "correlation": "positive",
            "message": "Higher confidence predictions are 5% more accurate"
        }
    """
    try:
        dashboard = _get_pg_accuracy()
        
        if not dashboard["ok"] or not dashboard.get("predictions"):
            return {
                "ok": True,
                "confidence_buckets": {},
                "correlation": "insufficient_data",
                "message": "No active predictions to analyze"
            }
        
        # Group predictions by confidence bucket
        buckets = defaultdict(lambda: {"predictions": [], "correct": 0, "total": 0})
        
        for pred in dashboard["predictions"]:
            confidence = pred.get("confidence", 0) * 100  # Convert to percentage
            is_correct = pred.get("is_correct_now", False)
            
            # Determine bucket (10% increments)
            bucket_start = int(confidence // 10) * 10
            bucket_end = bucket_start + 10
            bucket_key = f"{bucket_start}-{bucket_end}%"
            
            buckets[bucket_key]["predictions"].append(pred)
            buckets[bucket_key]["total"] += 1
            if is_correct:
                buckets[bucket_key]["correct"] += 1
        
        # Calculate accuracy per bucket
        bucket_stats = {}
        for bucket, data in buckets.items():
            accuracy = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0.0
            bucket_stats[bucket] = {
                "count": data["total"],
                "accuracy": accuracy,
                "correct": data["correct"]
            }
        
        # Determine correlation (simple heuristic: compare low vs high confidence)
        if len(bucket_stats) < 2:
            correlation = "insufficient_data"
            message = "Need predictions across multiple confidence levels"
        else:
            # Sort buckets by confidence
            sorted_buckets = sorted(bucket_stats.items(), key=lambda x: int(x[0].split('-')[0]))
            
            low_confidence_acc = sorted_buckets[0][1]["accuracy"]
            high_confidence_acc = sorted_buckets[-1][1]["accuracy"]
            
            diff = high_confidence_acc - low_confidence_acc
            
            if diff > 5.0:
                correlation = "positive"
                message = f"Higher confidence predictions are {diff:.1f}% more accurate"
            elif diff < -5.0:
                correlation = "negative"
                message = f"Lower confidence predictions are {abs(diff):.1f}% more accurate (unexpected!)"
            else:
                correlation = "neutral"
                message = f"Confidence and accuracy show weak correlation ({diff:.1f}% difference)"
        
        return {
            "ok": True,
            "confidence_buckets": bucket_stats,
            "correlation": correlation,
            "message": message,
            "total_predictions": dashboard["total_predictions"]
        }
        
    except Exception as e:
        LOGGER.error(f"Confidence correlation failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "confidence_buckets": {}
        }


def check_accuracy_alerts(threshold: float = 70.0) -> Dict[str, Any]:
    """
    Check if accuracy has dropped below threshold.
    
    Args:
        threshold: Accuracy percentage threshold (default 70%)
        
    Returns:
        {
            "ok": true,
            "alert": true/false,
            "current_accuracy": 65.0,
            "threshold": 70.0,
            "message": "⚠️ Accuracy dropped below 70% (currently 65%)",
            "symbols_affected": ["BTC", "ETH"]
        }
    """
    try:
        dashboard = _get_pg_accuracy()
        
        if not dashboard["ok"]:
            return {
                "ok": False,
                "alert": False,
                "error": dashboard.get("error", "Unknown error")
            }
        
        current_acc = dashboard["current_accuracy_pct"]
        alert_triggered = current_acc < threshold and dashboard["total_predictions"] > 0
        
        # Find which symbols are incorrect
        wrong_symbols = [
            pred["symbol"] for pred in dashboard["predictions"]
            if not pred.get("is_correct_now", False)
        ]
        
        if alert_triggered:
            message = f"⚠️ Accuracy dropped below {threshold}% (currently {current_acc:.1f}%)"
        else:
            message = f"✅ Accuracy is healthy: {current_acc:.1f}% (threshold: {threshold}%)"
        
        return {
            "ok": True,
            "alert": alert_triggered,
            "current_accuracy": current_acc,
            "threshold": threshold,
            "message": message,
            "symbols_affected": wrong_symbols,
            "wrong_count": len(wrong_symbols),
            "total_predictions": dashboard["total_predictions"]
        }
        
    except Exception as e:
        LOGGER.error(f"Accuracy alert check failed: {e}", exc_info=True)
        return {
            "ok": False,
            "alert": False,
            "error": str(e)
        }
