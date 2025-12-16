#!/usr/bin/env python3
"""
Live Accuracy Dashboard API
============================
Real-time accuracy tracking for all predictions before 48h evaluation.

Compares Ghost's predictions against current live prices to show
how predictions are performing in real-time.
"""

import logging
import time
from typing import Any, Dict, List
import requests
from core.prediction_store import get_prediction_store

LOGGER = logging.getLogger("ghost.live_accuracy")


def get_live_accuracy_dashboard() -> Dict[str, Any]:
    """
    Get real-time accuracy for all active predictions.
    
    Returns:
        {
            "ok": true,
            "current_accuracy_pct": 90.0,
            "total_predictions": 10,
            "correct_now": 9,
            "wrong_now": 1,
            "predictions": [...]
        }
    """
    try:
        store = get_prediction_store()
        
        # Get all recent predictions (last 48 hours, not yet evaluated)
        all_predictions = store.get_recent_predictions(limit=100)
        
        if not all_predictions:
            return {
                "ok": True,
                "current_accuracy_pct": 0.0,
                "total_predictions": 0,
                "correct_now": 0,
                "wrong_now": 0,
                "predictions": [],
                "message": "No active predictions found"
            }
        
        # Filter to only crypto symbols with Coinbase pricing
        crypto_symbols = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "DOT", "MATIC", "LTC", 
                         "LINK", "AVAX", "ATOM", "UNI", "XLM", "ALGO", "VET", "ICP", "FIL", "HBAR"]
        
        predictions_with_status = []
        correct_count = 0
        wrong_count = 0
        
        for pred in all_predictions:
            symbol = pred.get("symbol")
            if symbol not in crypto_symbols:
                continue
            
            try:
                # Get current live price
                price_response = requests.get(
                    f"https://api.coinbase.com/v2/prices/{symbol}-USD/spot",
                    timeout=3
                ).json()
                current_price = float(price_response['data']['amount'])
            except:
                # Skip if price unavailable
                continue
            
            pred_price = pred.get("entry_price") or pred.get("price_at_prediction")
            direction = pred.get("direction")
            confidence = pred.get("confidence", 0)
            pred_time = pred.get("run_at") or pred.get("created_at")
            
            if not pred_price or not direction or not pred_time:
                continue
            
            # Calculate price change
            price_change_pct = ((current_price - pred_price) / pred_price) * 100
            
            # Determine if currently correct
            is_correct = (
                (direction == "DOWN" and price_change_pct < 0) or
                (direction == "UP" and price_change_pct > 0)
            )
            
            if is_correct:
                correct_count += 1
                status = "✅ CORRECT"
            else:
                wrong_count += 1
                status = "❌ WRONG"
            
            # Calculate age
            age_hours = (time.time() - pred_time) / 3600
            hours_until_eval = 48 - age_hours
            
            predictions_with_status.append({
                "prediction_id": pred.get("prediction_id") or pred.get("id"),
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
                "entry_price": pred_price,
                "current_price": current_price,
                "price_change_pct": price_change_pct,
                "is_correct_now": is_correct,
                "status": status,
                "age_hours": age_hours,
                "hours_until_eval": max(0, hours_until_eval)
            })
        
        total = len(predictions_with_status)
        current_accuracy = (correct_count / total * 100) if total > 0 else 0.0
        
        # Sort by age (oldest first)
        predictions_with_status.sort(key=lambda x: x['age_hours'], reverse=True)
        
        return {
            "ok": True,
            "current_accuracy_pct": current_accuracy,
            "total_predictions": total,
            "correct_now": correct_count,
            "wrong_now": wrong_count,
            "predictions": predictions_with_status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        LOGGER.error(f"Live accuracy dashboard failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "current_accuracy_pct": 0.0,
            "total_predictions": 0,
            "correct_now": 0,
            "wrong_now": 0,
            "predictions": []
        }


def get_live_accuracy_by_symbol(symbol: str) -> Dict[str, Any]:
    """
    Get real-time accuracy for a specific symbol.
    
    Args:
        symbol: Trading symbol (e.g., "BTC", "ETH")
        
    Returns:
        Live accuracy details for that symbol
    """
    try:
        dashboard = get_live_accuracy_dashboard()
        
        if not dashboard["ok"]:
            return dashboard
        
        # Filter to specific symbol
        symbol_predictions = [
            p for p in dashboard["predictions"]
            if p["symbol"] == symbol
        ]
        
        if not symbol_predictions:
            return {
                "ok": True,
                "symbol": symbol,
                "current_accuracy_pct": 0.0,
                "total_predictions": 0,
                "correct_now": 0,
                "wrong_now": 0,
                "predictions": [],
                "message": f"No active predictions for {symbol}"
            }
        
        correct = sum(1 for p in symbol_predictions if p["is_correct_now"])
        wrong = len(symbol_predictions) - correct
        accuracy = (correct / len(symbol_predictions) * 100) if symbol_predictions else 0.0
        
        return {
            "ok": True,
            "symbol": symbol,
            "current_accuracy_pct": accuracy,
            "total_predictions": len(symbol_predictions),
            "correct_now": correct,
            "wrong_now": wrong,
            "predictions": symbol_predictions,
            "timestamp": time.time()
        }
        
    except Exception as e:
        LOGGER.error(f"Live accuracy by symbol failed for {symbol}: {e}", exc_info=True)
        return {
            "ok": False,
            "symbol": symbol,
            "error": str(e),
            "current_accuracy_pct": 0.0,
            "total_predictions": 0,
            "correct_now": 0,
            "wrong_now": 0,
            "predictions": []
        }
