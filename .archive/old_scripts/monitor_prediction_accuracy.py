#!/usr/bin/env python3
"""
Ghost Protocol - Prediction Accuracy Monitor
=============================================

Tracks actual predictions vs outcomes to:
1. Calculate real-world accuracy
2. Detect model drift
3. Trigger retraining when needed

Run daily: python monitor_prediction_accuracy.py

Author: Ghost AI
Date: December 19, 2025
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Storage
DATA_DIR = Path(__file__).parent / "data" / "accuracy_tracking"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_prediction_outcomes_from_db() -> list[dict]:
    """
    Fetch reconciled predictions from PostgreSQL.
    Returns predictions with actual outcomes.
    """
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            logger.warning("DATABASE_URL not set, using local tracking only")
            return []
        
        from sqlalchemy import create_engine, text
        engine = create_engine(database_url)
        
        query = text("""
            SELECT 
                id,
                symbol,
                direction as predicted_direction,
                actual_direction,
                confidence,
                was_correct,
                run_at,
                actual_price,
                features_json
            FROM ghost_predictions
            WHERE actual_direction IS NOT NULL
              AND was_correct IS NOT NULL
            ORDER BY run_at DESC
            LIMIT 1000
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            
            predictions = []
            for row in rows:
                predictions.append({
                    "id": row.id,
                    "symbol": row.symbol,
                    "predicted_direction": row.predicted_direction,
                    "actual_direction": row.actual_direction,
                    "confidence": float(row.confidence) if row.confidence else 0,
                    "was_correct": bool(row.was_correct),
                    "run_at": row.run_at.isoformat() if row.run_at else None,
                    "actual_price": float(row.actual_price) if row.actual_price else None,
                })
            
            return predictions
            
    except Exception as e:
        logger.error(f"Failed to fetch predictions from DB: {e}")
        return []


def calculate_accuracy_metrics(predictions: list[dict]) -> dict:
    """
    Calculate comprehensive accuracy metrics.
    """
    if not predictions:
        return {"error": "No predictions to analyze"}
    
    total = len(predictions)
    correct = sum(1 for p in predictions if p.get("was_correct"))
    
    # By confidence bucket
    confidence_buckets = {
        "high_90+": {"total": 0, "correct": 0},
        "good_80_90": {"total": 0, "correct": 0},
        "medium_70_80": {"total": 0, "correct": 0},
        "low_60_70": {"total": 0, "correct": 0},
        "very_low_<60": {"total": 0, "correct": 0},
    }
    
    for p in predictions:
        conf = p.get("confidence", 0) * 100
        was_correct = p.get("was_correct", False)
        
        if conf >= 90:
            bucket = "high_90+"
        elif conf >= 80:
            bucket = "good_80_90"
        elif conf >= 70:
            bucket = "medium_70_80"
        elif conf >= 60:
            bucket = "low_60_70"
        else:
            bucket = "very_low_<60"
        
        confidence_buckets[bucket]["total"] += 1
        if was_correct:
            confidence_buckets[bucket]["correct"] += 1
    
    # Calculate accuracy per bucket
    for bucket, data in confidence_buckets.items():
        if data["total"] > 0:
            data["accuracy"] = round(data["correct"] / data["total"] * 100, 1)
        else:
            data["accuracy"] = 0
    
    # By direction
    up_predictions = [p for p in predictions if p.get("predicted_direction") == "UP"]
    down_predictions = [p for p in predictions if p.get("predicted_direction") == "DOWN"]
    
    up_correct = sum(1 for p in up_predictions if p.get("was_correct"))
    down_correct = sum(1 for p in down_predictions if p.get("was_correct"))
    
    # By symbol (top 10)
    symbol_stats = {}
    for p in predictions:
        symbol = p.get("symbol", "UNKNOWN")
        if symbol not in symbol_stats:
            symbol_stats[symbol] = {"total": 0, "correct": 0}
        symbol_stats[symbol]["total"] += 1
        if p.get("was_correct"):
            symbol_stats[symbol]["correct"] += 1
    
    for symbol, data in symbol_stats.items():
        data["accuracy"] = round(data["correct"] / data["total"] * 100, 1) if data["total"] > 0 else 0
    
    # Sort by total predictions
    top_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:10]
    
    return {
        "overall": {
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy_pct": round(correct / total * 100, 1) if total > 0 else 0,
        },
        "by_direction": {
            "UP": {
                "total": len(up_predictions),
                "correct": up_correct,
                "accuracy_pct": round(up_correct / len(up_predictions) * 100, 1) if up_predictions else 0,
            },
            "DOWN": {
                "total": len(down_predictions),
                "correct": down_correct,
                "accuracy_pct": round(down_correct / len(down_predictions) * 100, 1) if down_predictions else 0,
            },
        },
        "by_confidence": confidence_buckets,
        "top_symbols": dict(top_symbols),
        "calculated_at": datetime.now().isoformat(),
    }


def check_model_drift(current_accuracy: float, historical_accuracy: float = 80.0) -> dict:
    """
    Detect if model accuracy has drifted significantly.
    Triggers retraining alert if accuracy drops below threshold.
    """
    drift_threshold = 10.0  # % points below expected
    retrain_threshold = 60.0  # Absolute minimum before retrain
    
    drift = historical_accuracy - current_accuracy
    
    status = "OK"
    action = None
    
    if current_accuracy < retrain_threshold:
        status = "CRITICAL"
        action = "RETRAIN_IMMEDIATELY"
    elif drift > drift_threshold:
        status = "WARNING"
        action = "SCHEDULE_RETRAIN"
    elif drift > 5.0:
        status = "MONITOR"
        action = "INCREASE_MONITORING"
    
    return {
        "current_accuracy": current_accuracy,
        "expected_accuracy": historical_accuracy,
        "drift_pct": round(drift, 1),
        "status": status,
        "action": action,
    }


def send_alert(message: str, level: str = "INFO"):
    """
    Send alert via Telegram if configured.
    """
    try:
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not bot_token or not chat_id:
            logger.warning("Telegram not configured, logging alert only")
            logger.info(f"[{level}] {message}")
            return
        
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(level, "📊")
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": f"{emoji} *Ghost AI Accuracy Monitor*\n\n{message}",
            "parse_mode": "Markdown",
        }
        
        requests.post(url, json=payload, timeout=10)
        
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")


def save_daily_report(metrics: dict, drift_status: dict):
    """
    Save daily accuracy report to file.
    """
    report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "metrics": metrics,
        "drift_status": drift_status,
    }
    
    filename = DATA_DIR / f"accuracy_report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved: {filename}")
    return filename


def run_accuracy_monitor():
    """
    Main monitoring function - run daily.
    """
    logger.info("=" * 60)
    logger.info("🔍 GHOST AI ACCURACY MONITOR")
    logger.info("=" * 60)
    
    # Fetch predictions
    predictions = fetch_prediction_outcomes_from_db()
    
    if not predictions:
        logger.warning("No predictions found to analyze")
        return
    
    logger.info(f"Analyzing {len(predictions)} reconciled predictions...")
    
    # Calculate metrics
    metrics = calculate_accuracy_metrics(predictions)
    
    # Check for drift
    current_accuracy = metrics["overall"]["accuracy_pct"]
    drift_status = check_model_drift(current_accuracy, historical_accuracy=80.0)
    
    # Print report
    logger.info("\n📊 ACCURACY REPORT")
    logger.info("-" * 40)
    logger.info(f"Overall Accuracy: {metrics['overall']['accuracy_pct']}%")
    logger.info(f"Total Predictions: {metrics['overall']['total_predictions']}")
    logger.info(f"Correct: {metrics['overall']['correct_predictions']}")
    
    logger.info("\n📈 BY DIRECTION:")
    for direction, data in metrics["by_direction"].items():
        logger.info(f"  {direction}: {data['accuracy_pct']}% ({data['correct']}/{data['total']})")
    
    logger.info("\n📊 BY CONFIDENCE:")
    for bucket, data in metrics["by_confidence"].items():
        if data["total"] > 0:
            logger.info(f"  {bucket}: {data['accuracy']}% ({data['correct']}/{data['total']})")
    
    logger.info(f"\n🔄 DRIFT STATUS: {drift_status['status']}")
    logger.info(f"  Expected: {drift_status['expected_accuracy']}%")
    logger.info(f"  Current: {drift_status['current_accuracy']}%")
    logger.info(f"  Drift: {drift_status['drift_pct']:+.1f}%")
    
    if drift_status["action"]:
        logger.info(f"  Action: {drift_status['action']}")
    
    # Save report
    save_daily_report(metrics, drift_status)
    
    # Send alerts if needed
    if drift_status["status"] == "CRITICAL":
        send_alert(
            f"🚨 Model accuracy dropped to {current_accuracy}%!\n"
            f"Expected: 80%\n"
            f"Action: Retrain immediately\n"
            f"Run: python train_ml_models_v2.py",
            level="CRITICAL"
        )
    elif drift_status["status"] == "WARNING":
        send_alert(
            f"⚠️ Model drift detected\n"
            f"Current: {current_accuracy}%\n"
            f"Expected: 80%\n"
            f"Drift: {drift_status['drift_pct']:+.1f}%\n"
            f"Schedule retraining within 7 days",
            level="WARNING"
        )
    
    return metrics, drift_status


if __name__ == "__main__":
    run_accuracy_monitor()
