#!/usr/bin/env python3
"""
Postgres-based Accuracy Calculator
===================================
Reads from ghost_prediction_outcomes table (persistent Postgres storage)
instead of ghost_predictions table (ephemeral SQLite).

This prevents accuracy data loss on Railway deployments.
"""

import logging
import os
import time
from typing import Any, Dict, List
import psycopg2
from psycopg2.extras import RealDictCursor

LOGGER = logging.getLogger("ghost.postgres_accuracy")


def calculate_accuracy_postgres(period: str = "all") -> Dict[str, Any]:
    """
    Calculate Ghost's prediction accuracy from Postgres.
    
    Args:
        period: 'all', '24h', '7d', '30d'
        
    Returns:
        Accuracy statistics dict
    """
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            LOGGER.error("DATABASE_URL not set")
            return _empty_response(period, "DATABASE_URL not configured")
        
        conn = psycopg2.connect(database_url)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Time filter
        time_filter = ""
        if period == "24h":
            cutoff_seconds = int(time.time()) - (24 * 3600)
            time_filter = f"AND EXTRACT(EPOCH FROM closed_at) >= {cutoff_seconds}"
        elif period == "7d":
            cutoff_seconds = int(time.time()) - (7 * 24 * 3600)
            time_filter = f"AND EXTRACT(EPOCH FROM closed_at) >= {cutoff_seconds}"
        elif period == "30d":
            cutoff_seconds = int(time.time()) - (30 * 24 * 3600)
            time_filter = f"AND EXTRACT(EPOCH FROM closed_at) >= {cutoff_seconds}"
        
        # Get completed outcomes
        cur.execute(f"""
            SELECT 
                prediction_id,
                symbol,
                closed_at,
                price_at_prediction,
                price_at_resolution,
                realized_move_pct,
                predicted_direction,
                actual_direction,
                hit_direction,
                predicted_confidence
            FROM ghost_prediction_outcomes
            WHERE status = 'completed' 
            AND hit_direction IS NOT NULL
            {time_filter}
            ORDER BY closed_at DESC
        """)
        
        rows = cur.fetchall()
        conn.close()
        
        if not rows:
            LOGGER.info(f"No completed outcomes found for period: {period}")
            return _empty_response(period, "No predictions evaluated yet")
        
        total = len(rows)
        correct = sum(1 for row in rows if row['hit_direction'] == 1)
        accuracy_pct = (correct / total) * 100 if total > 0 else 0.0
        
        # Calculate average error (absolute realized move vs predicted direction)
        total_error = 0.0
        for row in rows:
            realized_pct = abs(row['realized_move_pct'])
            total_error += realized_pct
        
        avg_error_pct = (total_error / total) if total > 0 else 0.0
        
        # Build predictions list
        predictions = []
        for row in rows:
            predictions.append({
                "prediction_id": row['prediction_id'],
                "symbol": row['symbol'],
                "closed_at": row['closed_at'].isoformat() if row['closed_at'] else None,
                "price_at_prediction": row['price_at_prediction'],
                "price_at_resolution": row['price_at_resolution'],
                "realized_move_pct": row['realized_move_pct'],
                "predicted_direction": row['predicted_direction'],
                "actual_direction": row['actual_direction'],
                "correct": row['hit_direction'] == 1,
                "confidence": row['predicted_confidence']
            })
        
        LOGGER.info(
            f"📊 Postgres accuracy ({period}): {accuracy_pct:.1f}% "
            f"({correct}/{total} correct)"
        )
        
        return {
            "period": period,
            "total_predictions": total,
            "resolved_predictions": total,
            "correct_predictions": correct,
            "accuracy_pct": accuracy_pct,
            "avg_error_pct": avg_error_pct,
            "predictions": predictions,
            "data_source": "postgres_outcomes"
        }
        
    except psycopg2.Error as e:
        LOGGER.error(f"Postgres query failed: {e}", exc_info=True)
        return _empty_response(period, f"Database error: {str(e)}")
    except Exception as e:
        LOGGER.error(f"Accuracy calculation failed: {e}", exc_info=True)
        return _empty_response(period, f"Error: {str(e)}")


def _empty_response(period: str, message: str = "") -> Dict[str, Any]:
    """Return empty accuracy response."""
    return {
        "period": period,
        "total_predictions": 0,
        "resolved_predictions": 0,
        "correct_predictions": 0,
        "accuracy_pct": 0.0,
        "avg_error_pct": 0.0,
        "predictions": [],
        "data_source": "postgres_outcomes",
        "message": message
    }


def get_accuracy_stats_postgres(period: str = "24h") -> Dict[str, Any]:
    """
    Get accuracy stats for Telegram reports.
    Wrapper around calculate_accuracy_postgres().
    """
    stats = calculate_accuracy_postgres(period)
    
    # Transform to Telegram-compatible format
    return {
        "accuracy_pct": stats["accuracy_pct"],
        "total_predictions": stats["total_predictions"],
        "correct_predictions": stats["correct_predictions"],
        "avg_error_pct": stats["avg_error_pct"]
    }
