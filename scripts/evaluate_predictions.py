#!/usr/bin/env python3
"""
Ghost Protocol Prediction Outcome Evaluator
============================================

Evaluates prediction accuracy by comparing predictions with actual price movements.

Key Metrics:
- Direction Accuracy: % of correct UP/DOWN predictions
- MAE (Mean Absolute Error): Average confidence error
- RMSE (Root Mean Squared Error): Confidence prediction quality
- Provider Performance: Accuracy by data source

Runs automatically or on-demand to populate outcomes table.
"""

import sys
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.providers.turbo_provider import turbo_stock_price, turbo_crypto_price
except ImportError:
    print("⚠️  Could not import turbo providers, running in standalone mode")
    turbo_stock_price = None
    turbo_crypto_price = None


class PredictionEvaluator:
    """Evaluates prediction accuracy against actual outcomes"""
    
    def __init__(self, db_path: str = "./data/ghost_predictions.db"):
        """Initialize evaluator with database connection"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_outcomes_table()
    
    def _ensure_outcomes_table(self):
        """Ensure outcomes table exists with correct schema"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                predicted_direction TEXT NOT NULL,
                actual_direction TEXT NOT NULL,
                predicted_confidence REAL NOT NULL,
                actual_price_change_pct REAL NOT NULL,
                was_correct INTEGER NOT NULL,
                confidence_error REAL NOT NULL,
                evaluated_at INTEGER NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcomes_symbol ON outcomes(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcomes_evaluated_at ON outcomes(evaluated_at)
        """)
        self.conn.commit()
    
    def get_expired_predictions(self, lookback_hours: int = 48) -> List[Dict]:
        """
        Get predictions that have expired (horizon passed) but not yet evaluated.
        
        Args:
            lookback_hours: How far back to look for expired predictions
            
        Returns:
            List of prediction dicts ready for evaluation
        """
        cursor = self.conn.cursor()
        now_ms = int(time.time() * 1000)
        lookback_ms = now_ms - (lookback_hours * 3600 * 1000)
        
        # Get predictions where:
        # 1. Created > lookback_hours ago
        # 2. Horizon has expired (created_at + horizon_h * 3600 * 1000 < now)
        # 3. Not yet evaluated (no outcome record)
        cursor.execute("""
            SELECT p.*
            FROM predictions p
            LEFT JOIN outcomes o ON p.id = o.prediction_id
            WHERE p.run_at > ?
              AND p.run_at + (p.horizon_h * 3600 * 1000) < ?
              AND o.id IS NULL
            ORDER BY p.run_at ASC
            LIMIT 100
        """, (lookback_ms, now_ms))
        
        predictions = []
        for row in cursor.fetchall():
            predictions.append({
                "id": row["id"],
                "symbol": row["symbol"],
                "asset_type": row["asset_type"],
                "direction": row["direction"],
                "confidence": row["confidence"],
                "current_price": row["current_price"],
                "run_at": row["run_at"],
                "horizon_h": row["horizon_h"],
            })
        
        return predictions
    
    def get_current_price(self, symbol: str, asset_type: str) -> Optional[float]:
        """
        Fetch current price for symbol.
        
        Args:
            symbol: Ticker symbol
            asset_type: "stock" or "crypto"
            
        Returns:
            Current price or None if fetch failed
        """
        try:
            if asset_type == "crypto" and turbo_crypto_price:
                result = turbo_crypto_price(symbol, max_budget_s=3.0)
                if result["ok"]:
                    return result["price"]
            elif asset_type == "stock" and turbo_stock_price:
                result = turbo_stock_price(symbol, max_budget_s=3.0)
                if result["ok"]:
                    return result["price"]
            return None
        except Exception as e:
            print(f"❌ Failed to fetch price for {symbol}: {e}")
            return None
    
    def evaluate_prediction(self, prediction: Dict) -> Optional[Dict]:
        """
        Evaluate a single prediction against actual outcome.
        
        Args:
            prediction: Prediction dict with id, symbol, direction, etc.
            
        Returns:
            Outcome dict or None if evaluation failed
        """
        # Get current price
        current_price = self.get_current_price(
            prediction["symbol"],
            prediction["asset_type"]
        )
        
        if current_price is None:
            print(f"⚠️  Could not fetch current price for {prediction['symbol']}, skipping")
            return None
        
        # Calculate actual price change
        original_price = prediction["current_price"]
        price_change_pct = ((current_price - original_price) / original_price) * 100
        
        # Determine actual direction
        actual_direction = "UP" if price_change_pct > 0 else "DOWN"
        
        # Check if prediction was correct
        was_correct = (prediction["direction"] == actual_direction)
        
        # Calculate confidence error
        # Ideal: confidence should match abs(price_change_pct) / expected_max_change
        # For simplicity: error = abs(confidence - (1 if correct else 0))
        confidence_error = abs(prediction["confidence"] - (1.0 if was_correct else 0.0))
        
        outcome = {
            "prediction_id": prediction["id"],
            "symbol": prediction["symbol"],
            "predicted_direction": prediction["direction"],
            "actual_direction": actual_direction,
            "predicted_confidence": prediction["confidence"],
            "actual_price_change_pct": price_change_pct,
            "was_correct": 1 if was_correct else 0,
            "confidence_error": confidence_error,
            "evaluated_at": int(time.time() * 1000),
        }
        
        return outcome
    
    def save_outcome(self, outcome: Dict):
        """Save outcome to database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO outcomes (
                prediction_id, symbol, predicted_direction, actual_direction,
                predicted_confidence, actual_price_change_pct, was_correct,
                confidence_error, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            outcome["prediction_id"],
            outcome["symbol"],
            outcome["predicted_direction"],
            outcome["actual_direction"],
            outcome["predicted_confidence"],
            outcome["actual_price_change_pct"],
            outcome["was_correct"],
            outcome["confidence_error"],
            outcome["evaluated_at"],
        ))
        self.conn.commit()
    
    def evaluate_all_expired(self) -> Dict:
        """
        Evaluate all expired predictions.
        
        Returns:
            Summary dict with counts and metrics
        """
        predictions = self.get_expired_predictions()
        
        if not predictions:
            return {
                "evaluated": 0,
                "message": "No expired predictions to evaluate"
            }
        
        evaluated = 0
        correct = 0
        total_confidence_error = 0.0
        
        print(f"\n🔍 Evaluating {len(predictions)} expired predictions...")
        
        for i, pred in enumerate(predictions, 1):
            outcome = self.evaluate_prediction(pred)
            
            if outcome:
                self.save_outcome(outcome)
                evaluated += 1
                correct += outcome["was_correct"]
                total_confidence_error += outcome["confidence_error"]
                
                status = "✅" if outcome["was_correct"] else "❌"
                print(f"{status} [{i}/{len(predictions)}] {pred['symbol']}: "
                      f"Predicted {pred['direction']}, "
                      f"Actual {outcome['actual_direction']} "
                      f"({outcome['actual_price_change_pct']:+.2f}%)")
            
            # Sleep briefly to avoid rate limits
            time.sleep(0.5)
        
        accuracy = (correct / evaluated * 100) if evaluated > 0 else 0
        avg_confidence_error = (total_confidence_error / evaluated) if evaluated > 0 else 0
        
        summary = {
            "evaluated": evaluated,
            "correct": correct,
            "accuracy": accuracy,
            "avg_confidence_error": avg_confidence_error,
        }
        
        print(f"\n📊 Evaluation Complete:")
        print(f"   Evaluated: {evaluated}/{len(predictions)}")
        print(f"   Correct: {correct}/{evaluated} ({accuracy:.1f}%)")
        print(f"   Avg Confidence Error: {avg_confidence_error:.3f}")
        
        return summary
    
    def get_accuracy_report(self, days: int = 7) -> Dict:
        """
        Get accuracy report for last N days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Report dict with overall and per-symbol accuracy
        """
        cursor = self.conn.cursor()
        cutoff_ms = int((time.time() - days * 86400) * 1000)
        
        # Overall accuracy
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(was_correct) as correct,
                AVG(confidence_error) as avg_error
            FROM outcomes
            WHERE evaluated_at > ?
        """, (cutoff_ms,))
        
        row = cursor.fetchone()
        overall = {
            "total": row["total"],
            "correct": row["correct"],
            "accuracy": (row["correct"] / row["total"] * 100) if row["total"] > 0 else 0,
            "avg_confidence_error": row["avg_error"] or 0,
        }
        
        # Per-symbol accuracy
        cursor.execute("""
            SELECT 
                symbol,
                COUNT(*) as total,
                SUM(was_correct) as correct,
                AVG(confidence_error) as avg_error
            FROM outcomes
            WHERE evaluated_at > ?
            GROUP BY symbol
            ORDER BY total DESC
        """, (cutoff_ms,))
        
        by_symbol = []
        for row in cursor.fetchall():
            by_symbol.append({
                "symbol": row["symbol"],
                "total": row["total"],
                "correct": row["correct"],
                "accuracy": (row["correct"] / row["total"] * 100) if row["total"] > 0 else 0,
                "avg_confidence_error": row["avg_error"] or 0,
            })
        
        return {
            "overall": overall,
            "by_symbol": by_symbol,
            "period_days": days,
        }


def main():
    """Run prediction evaluation"""
    print("=" * 60)
    print("GHOST PROTOCOL PREDICTION EVALUATOR")
    print("=" * 60)
    
    evaluator = PredictionEvaluator()
    
    # Evaluate all expired predictions
    summary = evaluator.evaluate_all_expired()
    
    # Show accuracy report
    if summary["evaluated"] > 0:
        print("\n" + "=" * 60)
        report = evaluator.get_accuracy_report(days=7)
        print(f"\n📈 7-Day Accuracy Report:")
        print(f"   Overall: {report['overall']['correct']}/{report['overall']['total']} "
              f"({report['overall']['accuracy']:.1f}%)")
        
        if report['by_symbol']:
            print(f"\n   Top Symbols:")
            for sym in report['by_symbol'][:10]:
                print(f"   - {sym['symbol']}: {sym['correct']}/{sym['total']} "
                      f"({sym['accuracy']:.1f}%)")


if __name__ == "__main__":
    main()
