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
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "evaluator.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger(__name__)

try:
    from core.providers.turbo_provider import turbo_stock_price, turbo_crypto_price
except ImportError:
    LOGGER.warning("Could not import turbo providers, running in standalone mode")
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
    
    def get_expired_predictions(self, lookback_hours: int = 168) -> List[Dict]:
        """
        Get predictions that have expired (horizon passed) but not yet evaluated.
        
        Args:
            lookback_hours: How far back to look for expired predictions
            
        Returns:
            List of prediction dicts ready for evaluation
        """
        cursor = self.conn.cursor()
        now_sec = time.time()
        lookback_sec = now_sec - (lookback_hours * 3600)
        
        # Get predictions where:
        # 1. Created > lookback_hours ago
        # 2. Horizon has expired (run_at + horizon_h * 3600 < now)
        # 3. Not yet evaluated (no outcome record)
        cursor.execute("""
            SELECT p.id, p.symbol, p.direction, p.confidence, p.run_at, p.horizon_h,
                   pp.price as original_price
            FROM predictions p
            LEFT JOIN outcomes o ON p.id = o.prediction_id
            LEFT JOIN prediction_points pp ON p.id = pp.prediction_id AND pp.kind = 'forecast'
            WHERE p.run_at > ?
              AND p.run_at + (p.horizon_h * 3600) < ?
              AND o.id IS NULL
              AND pp.ts = (SELECT MIN(ts) FROM prediction_points WHERE prediction_id = p.id AND kind = 'forecast')
            GROUP BY p.id
            ORDER BY p.run_at ASC
            LIMIT 100
        """, (lookback_sec, now_sec))
        
        predictions = []
        # Expanded crypto symbols list from DEFAULT_CRYPTO_SYMBOLS (52 coins)
        crypto_symbols = {
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'AVAX',
            'DOT', 'MATIC', 'SHIB', 'LTC', 'UNI', 'LINK', 'ATOM', 'ETC',
            'PEPE', 'ARB', 'OP', 'INJ', 'TIA', 'SUI', 'APT', 'SEI',
            'FTM', 'NEAR', 'ALGO', 'VET', 'FIL', 'AAVE', 'MKR', 'SNX',
            'COMP', 'CRV', '1INCH', 'BAL', 'SUSHI', 'YFI', 'LDO', 'RPL',
            'IMX', 'SAND', 'MANA', 'AXS', 'GALA', 'ENJ', 'CHZ', 'FLOW',
            'ICP', 'HBAR', 'QNT', 'RUNE',
            # Legacy VIP coins
            'WEPE', 'LILPEPE', 'DORKL', 'SLOTH', 'APC'
        }
        
        for row in cursor.fetchall():
            # Infer asset type from symbol
            asset_type = 'crypto' if row["symbol"] in crypto_symbols else 'stock'
            
            predictions.append({
                "id": row["id"],
                "symbol": row["symbol"],
                "asset_type": asset_type,
                "direction": row["direction"],
                "confidence": row["confidence"],
                "original_price": row["original_price"],
                "run_at": row["run_at"],
                "horizon_h": row["horizon_h"],
            })
        
        LOGGER.info(f"Found {len(predictions)} expired predictions to evaluate")
        return predictions
    
    def get_live_price(self, symbol: str, asset_type: str) -> Optional[float]:
        """
        Fetch current live price for symbol with robust error handling.
        
        Args:
            symbol: Ticker symbol
            asset_type: "stock" or "crypto"
            
        Returns:
            Current price or None if fetch failed
        """
        try:
            if asset_type == "crypto":
                # Try turbo provider first
                if turbo_crypto_price:
                    result = turbo_crypto_price(symbol, max_budget_s=3.0)
                    if result["ok"]:
                        LOGGER.debug(f"Fetched {symbol} crypto price: ${result['price']:.2f} (turbo)")
                        return result["price"]
                
                # Fallback to Coinbase API
                import requests
                url = f"https://api.coinbase.com/v2/prices/{symbol}-USD/spot"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    price = float(data["data"]["amount"])
                    LOGGER.debug(f"Fetched {symbol} crypto price: ${price:.2f} (Coinbase)")
                    return price
                else:
                    LOGGER.warning(f"Coinbase API returned {response.status_code} for {symbol}")
                    
            elif asset_type == "stock":
                # Try turbo provider first
                if turbo_stock_price:
                    result = turbo_stock_price(symbol, max_budget_s=3.0)
                    if result["ok"]:
                        LOGGER.debug(f"Fetched {symbol} stock price: ${result['price']:.2f} (turbo)")
                        return result["price"]
                
                # Fallback to yfinance with retry/backoff (robust pattern from wolf_app.py)
                price = self._fetch_yfinance_with_retry(symbol)
                if price:
                    LOGGER.debug(f"Fetched {symbol} stock price: ${price:.2f} (yfinance)")
                    return price
                    
            return None
            
        except Exception as e:
            LOGGER.error(f"Failed to fetch price for {symbol} ({asset_type}): {e}")
            return None
    
    def _fetch_yfinance_with_retry(self, symbol: str, max_retries: int = 3) -> Optional[float]:
        """
        Fetch stock price from yfinance with exponential backoff for JSON errors.
        Pattern from wolf_app.py line 9235.
        
        Args:
            symbol: Stock ticker symbol
            max_retries: Maximum number of retry attempts
            
        Returns:
            Current stock price or None if fetch failed
        """
        base_delay = 0.5  # Start with 500ms
        
        for attempt in range(max_retries):
            try:
                import yfinance as yf
                
                # Increase timeout and add better JSON error handling
                tkr = yf.Ticker(symbol.upper())
                tkr.session.timeout = (5, 15)  # (connect, read) timeouts
                
                # Get recent price data
                hist = tkr.history(period="1d")
                if not hist.empty:
                    close = float(hist["Close"].iloc[-1])
                    if close > 0:
                        return close
                        
                LOGGER.warning(f"yfinance returned empty data for {symbol}")
                return None
                
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                
                # Check if it's a JSON parsing error (retryable)
                is_json_error = "expecting value" in low or "json" in low
                
                # Retry on JSON errors with exponential backoff
                if is_json_error and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # 0.5s, 1s, 2s
                    LOGGER.debug(
                        f"yfinance JSON error for {symbol}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue  # Retry
                
                # Not retryable or final attempt - log and fail
                LOGGER.warning(f"yfinance failed for {symbol} after {attempt + 1} attempts: {msg}")
                return None
        
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
        current_price = self.get_live_price(
            prediction["symbol"],
            prediction["asset_type"]
        )
        
        if current_price is None:
            LOGGER.warning(
                f"⚠️  {prediction['symbol']}: No live price available from any provider. "
                f"Asset type: {prediction['asset_type']}. Skipping evaluation."
            )
            return None
        
        # Calculate actual price change
        original_price = prediction["original_price"]
        if original_price is None or original_price <= 0:
            LOGGER.warning(
                f"⚠️  {prediction['symbol']}: Invalid original price ({original_price}). "
                f"Skipping evaluation."
            )
            return None
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
            LOGGER.info("No expired predictions to evaluate")
            return {
                "evaluated": 0,
                "message": "No expired predictions to evaluate"
            }
        
        evaluated = 0
        skipped = 0
        correct = 0
        incorrect = 0
        total_confidence_error = 0.0
        
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"🔍 EVALUATING {len(predictions)} EXPIRED PREDICTIONS")
        LOGGER.info(f"{'='*60}\n")
        
        for i, pred in enumerate(predictions, 1):
            outcome = self.evaluate_prediction(pred)
            
            if outcome:
                self.save_outcome(outcome)
                evaluated += 1
                
                if outcome["was_correct"]:
                    correct += 1
                    status = "✅ CORRECT"
                else:
                    incorrect += 1
                    status = "❌ INCORRECT"
                    
                total_confidence_error += outcome["confidence_error"]
                
                LOGGER.info(
                    f"{status} [{i}/{len(predictions)}] {pred['symbol']} ({pred['asset_type']}): "
                    f"Predicted {pred['direction']}, "
                    f"Actual {outcome['actual_direction']} "
                    f"({outcome['actual_price_change_pct']:+.2f}%) | "
                    f"Confidence: {pred['confidence']:.2%}"
                )
            else:
                skipped += 1
                LOGGER.warning(
                    f"⏭️  SKIPPED [{i}/{len(predictions)}] {pred['symbol']} ({pred['asset_type']}): "
                    f"Could not fetch live price (see warnings above)"
                )
            
            # Sleep briefly to avoid rate limits
            time.sleep(0.5)
        
        accuracy = (correct / evaluated * 100) if evaluated > 0 else 0
        avg_confidence_error = (total_confidence_error / evaluated) if evaluated > 0 else 0
        
        summary = {
            "total_expired": len(predictions),
            "evaluated": evaluated,
            "skipped": skipped,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy,
            "avg_confidence_error": avg_confidence_error,
        }
        
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info("📊 EVALUATION COMPLETE")
        LOGGER.info(f"{'='*60}")
        LOGGER.info(f"   Total Expired Predictions: {len(predictions)}")
        LOGGER.info(f"   Successfully Evaluated: {evaluated}")
        LOGGER.info(f"   Skipped (no price data): {skipped}")
        LOGGER.info(f"   Correct: {correct}/{evaluated} ({accuracy:.1f}%)")
        LOGGER.info(f"   Incorrect: {incorrect}/{evaluated}")
        LOGGER.info(f"   Avg Confidence Error: {avg_confidence_error:.3f}")
        LOGGER.info(f"{'='*60}\n")
        
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
    from datetime import datetime
    
    LOGGER.info("="*60)
    LOGGER.info("GHOST PROTOCOL PREDICTION EVALUATOR")
    LOGGER.info("="*60)
    LOGGER.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    LOGGER.info("Database: ./data/ghost_predictions.db")
    LOGGER.info("Log file: ./logs/evaluator.log\n")
    
    evaluator = PredictionEvaluator()
    
    # Evaluate all expired predictions
    summary = evaluator.evaluate_all_expired()
    
    # Show accuracy report
    if summary["evaluated"] > 0:
        LOGGER.info("\n" + "="*60)
        report = evaluator.get_accuracy_report(days=7)
        LOGGER.info("📈 7-DAY ACCURACY REPORT")
        LOGGER.info("="*60)
        LOGGER.info(f"   Overall: {report['overall']['correct']}/{report['overall']['total']} "
                   f"({report['overall']['accuracy']:.1f}%)")
        LOGGER.info(f"   Avg Confidence Error: {report['overall']['avg_confidence_error']:.3f}")
        
        if report['by_symbol']:
            LOGGER.info("\n   📊 Top Symbols by Volume:")
            for sym in report['by_symbol'][:15]:
                LOGGER.info(f"      {sym['symbol']}: {sym['correct']}/{sym['total']} "
                          f"({sym['accuracy']:.1f}%) | Error: {sym['avg_confidence_error']:.3f}")
    
    LOGGER.info(f"\n✅ Evaluation complete at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return summary for programmatic use
    return summary


if __name__ == "__main__":
    main()
