"""
Phase 1.3 & 6.5: Comprehensive Backtesting Framework

Validates prediction models on historical data before deployment.
Tests model on 90 days of historical market data to ensure accuracy.

Features:
- Load historical OHLCV data from multiple sources
- Generate predictions using current model
- Compare predictions against actual market movements
- Calculate accuracy, win rate, profit factor
- Identify best/worst performing symbols
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import get_logger
from core.db_pool import get_pool

LOGGER = get_logger(__name__)


class BacktestFramework:
    """
    Framework for backtesting prediction models on historical data.
    """
    
    def __init__(self):
        self.results = {
            "predictions": [],
            "summary": {},
            "by_symbol": {},
            "by_direction": {}
        }
    
    async def load_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Load historical OHLCV data for a symbol.
        
        Args:
            symbol: Ticker symbol
            start_date: Start of backtest period
            end_date: End of backtest period
        
        Returns:
            List of OHLCV candles with timestamps
        """
        try:
            # Try to load from database first
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        timestamp,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM ghost_ohlcv_data
                    WHERE symbol = $1
                    AND timestamp BETWEEN $2 AND $3
                    ORDER BY timestamp ASC
                """, symbol, start_date, end_date)
                
                if rows:
                    return [dict(row) for row in rows]
            
            # Fallback: Fetch from yfinance
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1h"
            )
            
            if df.empty:
                LOGGER.warning(f"No historical data for {symbol}")
                return []
            
            data = []
            for timestamp, row in df.iterrows():
                data.append({
                    "timestamp": timestamp,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"])
                })
            
            return data
            
        except Exception as e:
            LOGGER.error(f"Failed to load historical data for {symbol}: {e}")
            return []
    
    async def generate_prediction(
        self,
        symbol: str,
        timestamp: datetime,
        historical_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate prediction for a symbol at a specific timestamp.
        
        Args:
            symbol: Ticker symbol
            timestamp: When prediction is made
            historical_data: Historical OHLCV data up to timestamp
        
        Returns:
            {
                "symbol": symbol,
                "timestamp": timestamp,
                "direction": "UP" or "DOWN",
                "confidence": 0-100,
                "entry_price": float,
                "target_price": float,
                "stop_loss": float
            }
        """
        try:
            # Filter data up to prediction timestamp
            relevant_data = [
                d for d in historical_data 
                if d["timestamp"] <= timestamp
            ]
            
            if len(relevant_data) < 20:  # Need at least 20 candles
                return None
            
            # Use current stock prediction engine
            from core.stock_engine import predict_stock_direction
            
            # Get current price from last candle
            current_price = relevant_data[-1]["close"]
            
            # Generate prediction (simplified - in production use full feature set)
            result = await predict_stock_direction(
                symbol=symbol,
                current_price=current_price,
                lookback_candles=relevant_data[-60:]  # Last 60 candles
            )
            
            if not result or result.get("error"):
                return None
            
            return {
                "symbol": symbol,
                "timestamp": timestamp,
                "direction": result.get("direction"),
                "confidence": result.get("confidence", 0),
                "entry_price": current_price,
                "target_price": result.get("target_price"),
                "stop_loss": result.get("stop_loss")
            }
            
        except Exception as e:
            LOGGER.error(f"Failed to generate prediction for {symbol}: {e}")
            return None
    
    def evaluate_prediction(
        self,
        prediction: Dict[str, Any],
        actual_data: List[Dict[str, Any]],
        eval_hours: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate prediction against actual market movement.
        
        Args:
            prediction: Prediction dict from generate_prediction
            actual_data: Historical data including prediction timestamp
            eval_hours: Hours to wait before evaluating (default 1)
        
        Returns:
            {
                "correct": True/False,
                "actual_move_pct": float,
                "hit_target": True/False,
                "hit_stop": True/False,
                "max_favorable_pct": float,
                "max_adverse_pct": float
            }
        """
        try:
            pred_timestamp = prediction["timestamp"]
            entry_price = prediction["entry_price"]
            direction = prediction["direction"]
            target_price = prediction.get("target_price")
            stop_loss = prediction.get("stop_loss")
            
            # Find data after prediction
            future_data = [
                d for d in actual_data
                if d["timestamp"] > pred_timestamp
                and d["timestamp"] <= pred_timestamp + timedelta(hours=eval_hours)
            ]
            
            if not future_data:
                return {"correct": None, "error": "No future data available"}
            
            # Get final price
            final_price = future_data[-1]["close"]
            actual_move_pct = ((final_price - entry_price) / entry_price) * 100
            
            # Check if prediction was correct
            if direction == "UP":
                correct = actual_move_pct > 0
            else:  # DOWN
                correct = actual_move_pct < 0
            
            # Track max favorable/adverse moves
            max_favorable = 0.0
            max_adverse = 0.0
            hit_target = False
            hit_stop = False
            
            for candle in future_data:
                high = candle["high"]
                low = candle["low"]
                
                if direction == "UP":
                    move_high_pct = ((high - entry_price) / entry_price) * 100
                    move_low_pct = ((low - entry_price) / entry_price) * 100
                    max_favorable = max(max_favorable, move_high_pct)
                    max_adverse = min(max_adverse, move_low_pct)
                    
                    if target_price and high >= target_price:
                        hit_target = True
                    if stop_loss and low <= stop_loss:
                        hit_stop = True
                else:  # DOWN
                    move_low_pct = -((entry_price - low) / entry_price) * 100
                    move_high_pct = -((entry_price - high) / entry_price) * 100
                    max_favorable = max(max_favorable, move_low_pct)
                    max_adverse = min(max_adverse, move_high_pct)
                    
                    if target_price and low <= target_price:
                        hit_target = True
                    if stop_loss and high >= stop_loss:
                        hit_stop = True
            
            return {
                "correct": correct,
                "actual_move_pct": actual_move_pct,
                "hit_target": hit_target,
                "hit_stop": hit_stop,
                "max_favorable_pct": max_favorable,
                "max_adverse_pct": max_adverse
            }
            
        except Exception as e:
            LOGGER.error(f"Failed to evaluate prediction: {e}")
            return {"correct": None, "error": str(e)}
    
    async def run_backtest(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        prediction_interval_hours: int = 1
    ) -> Dict[str, Any]:
        """
        Run backtest on multiple symbols.
        
        Args:
            symbols: List of symbols to test
            start_date: Start of backtest period
            end_date: End of backtest period
            prediction_interval_hours: Hours between predictions
        
        Returns:
            Backtest results with summary statistics
        """
        LOGGER.info(f"Starting backtest: {len(symbols)} symbols from {start_date} to {end_date}")
        
        all_predictions = []
        
        for symbol in symbols:
            LOGGER.info(f"Backtesting {symbol}...")
            
            # Load historical data
            historical_data = await self.load_historical_data(symbol, start_date, end_date)
            
            if not historical_data:
                LOGGER.warning(f"No data for {symbol}, skipping")
                continue
            
            # Generate predictions at intervals
            current_time = start_date + timedelta(hours=24)  # Skip first day for warmup
            
            while current_time < end_date - timedelta(hours=prediction_interval_hours):
                prediction = await self.generate_prediction(symbol, current_time, historical_data)
                
                if prediction:
                    # Evaluate prediction
                    evaluation = self.evaluate_prediction(
                        prediction,
                        historical_data,
                        eval_hours=prediction_interval_hours
                    )
                    
                    prediction.update(evaluation)
                    all_predictions.append(prediction)
                
                current_time += timedelta(hours=prediction_interval_hours)
        
        # Calculate summary statistics
        self.results["predictions"] = all_predictions
        self.results["summary"] = self._calculate_summary(all_predictions)
        self.results["by_symbol"] = self._calculate_by_symbol(all_predictions)
        self.results["by_direction"] = self._calculate_by_direction(all_predictions)
        
        return self.results
    
    def _calculate_summary(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall summary statistics."""
        valid_predictions = [p for p in predictions if p.get("correct") is not None]
        
        if not valid_predictions:
            return {"error": "No valid predictions"}
        
        total = len(valid_predictions)
        correct = sum(1 for p in valid_predictions if p["correct"])
        accuracy = (correct / total) * 100
        
        avg_move = sum(abs(p.get("actual_move_pct", 0)) for p in valid_predictions) / total
        
        return {
            "total_predictions": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy_pct": round(accuracy, 2),
            "avg_move_pct": round(avg_move, 2)
        }
    
    def _calculate_by_symbol(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics grouped by symbol."""
        by_symbol = {}
        
        for pred in predictions:
            if pred.get("correct") is None:
                continue
            
            symbol = pred["symbol"]
            if symbol not in by_symbol:
                by_symbol[symbol] = {"total": 0, "correct": 0}
            
            by_symbol[symbol]["total"] += 1
            if pred["correct"]:
                by_symbol[symbol]["correct"] += 1
        
        for symbol, stats in by_symbol.items():
            stats["accuracy_pct"] = round((stats["correct"] / stats["total"]) * 100, 2)
        
        return by_symbol
    
    def _calculate_by_direction(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics grouped by direction."""
        by_direction = {"UP": {"total": 0, "correct": 0}, "DOWN": {"total": 0, "correct": 0}}
        
        for pred in predictions:
            if pred.get("correct") is None:
                continue
            
            direction = pred.get("direction", "HOLD")
            if direction in by_direction:
                by_direction[direction]["total"] += 1
                if pred["correct"]:
                    by_direction[direction]["correct"] += 1
        
        for direction, stats in by_direction.items():
            if stats["total"] > 0:
                stats["accuracy_pct"] = round((stats["correct"] / stats["total"]) * 100, 2)
            else:
                stats["accuracy_pct"] = 0
        
        return by_direction
    
    def save_results(self, output_path: str = "backtest_results.json"):
        """Save backtest results to file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        LOGGER.info(f"Backtest results saved to {output_path}")


async def run_90_day_backtest(symbols: Optional[List[str]] = None):
    """
    Phase 6.5: Run 90-day backtest on edge symbols.
    """
    if symbols is None:
        # Default to edge symbols
        symbols = ["BTC", "ETH", "XRP", "LINK", "CHZ", "AAPL", "NVDA", "PANW", "NET", "FTNT", "DDOG"]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    framework = BacktestFramework()
    results = await framework.run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        prediction_interval_hours=1
    )
    
    # Print summary
    print("\n" + "="*70)
    print("📊 90-DAY BACKTEST RESULTS")
    print("="*70)
    summary = results.get("summary", {})
    print(f"\nTotal Predictions: {summary.get('total_predictions', 0)}")
    print(f"Correct: {summary.get('correct', 0)}")
    print(f"Accuracy: {summary.get('accuracy_pct', 0):.2f}%")
    print(f"Avg Move: {summary.get('avg_move_pct', 0):.2f}%")
    
    print("\n📈 By Symbol:")
    by_symbol = results.get("by_symbol", {})
    for symbol, stats in sorted(by_symbol.items(), key=lambda x: x[1]["accuracy_pct"], reverse=True):
        print(f"  {symbol:6} {stats['accuracy_pct']:5.1f}% ({stats['correct']}/{stats['total']})")
    
    print("\n🎯 By Direction:")
    by_direction = results.get("by_direction", {})
    for direction, stats in by_direction.items():
        print(f"  {direction:4} {stats['accuracy_pct']:5.1f}% ({stats['correct']}/{stats['total']})")
    
    print("="*70 + "\n")
    
    # Save results
    framework.save_results("backtest_90d_results.json")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_90_day_backtest())
