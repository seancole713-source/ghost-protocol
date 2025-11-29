#!/usr/bin/env python3
"""
Ghost Protocol - Backtesting Engine with Walk-Forward Analysis
=============================================================
Validates strategies before live deployment

Prevents deploying untested strategies
"""

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Database path
DATA_DIR = Path(__file__).parent.parent / "data"
BACKTEST_DB = DATA_DIR / "backtest_results.db"


@dataclass
class BacktestTrade:
    """Single backtest trade"""
    entry_time: float
    exit_time: float
    symbol: str
    direction: str  # BUY/SELL
    entry_price: float
    exit_price: float
    shares: int
    pnl_dollar: float
    pnl_pct: float
    confidence: float
    was_correct: bool


@dataclass
class BacktestResult:
    """Complete backtest results"""
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: list[BacktestTrade]


class BacktestEngine:
    """Historical simulation with walk-forward analysis"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self._init_database()
        
    def _init_database(self):
        """Initialize backtest results database"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(BACKTEST_DB)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT,
                    start_date REAL,
                    end_date REAL,
                    total_trades INTEGER,
                    win_rate REAL,
                    total_return_pct REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    created_at REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id INTEGER,
                    entry_time REAL,
                    exit_time REAL,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    shares INTEGER,
                    pnl_dollar REAL,
                    pnl_pct REAL,
                    confidence REAL,
                    was_correct INTEGER,
                    FOREIGN KEY (backtest_id) REFERENCES backtest_results (id)
                )
            """)
            conn.commit()
    
    def run_backtest(
        self,
        predictions: list[dict[str, Any]],
        historical_prices: dict[str, list[dict]],
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """
        Run backtest simulation
        
        Args:
            predictions: List of prediction dicts with symbol, confidence, direction
            historical_prices: {symbol: [{time, price, ...}, ...]}
            start_date: Backtest start
            end_date: Backtest end
        
        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Running backtest: {start_date} → {end_date}")
        
        trades = []
        equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        open_positions = {}
        
        # Sort predictions by time
        predictions.sort(key=lambda x: x.get("timestamp", 0))
        
        for pred in predictions:
            symbol = pred.get("symbol")
            direction = pred.get("direction", "HOLD")
            confidence = pred.get("confidence", 0.5)
            entry_time = pred.get("timestamp", time.time())
            
            if direction == "HOLD" or confidence < 0.6:
                continue
            
            # Get historical price at entry time
            prices = historical_prices.get(symbol, [])
            entry_price = self._get_price_at_time(prices, entry_time)
            
            if not entry_price or entry_price <= 0:
                continue
            
            # Calculate position size (simple 5% of capital)
            position_size = current_capital * 0.05
            shares = int(position_size / entry_price)
            
            if shares == 0:
                continue
            
            # Hold position for 24 hours
            exit_time = entry_time + (24 * 3600)
            exit_price = self._get_price_at_time(prices, exit_time)
            
            if not exit_price or exit_price <= 0:
                exit_price = entry_price  # No change
            
            # Calculate P&L
            if direction == "BUY":
                pnl_dollar = (exit_price - entry_price) * shares
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                was_correct = exit_price > entry_price
            else:  # SELL
                pnl_dollar = (entry_price - exit_price) * shares
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                was_correct = exit_price < entry_price
            
            current_capital += pnl_dollar
            equity_curve.append(current_capital)
            
            trade = BacktestTrade(
                entry_time=entry_time,
                exit_time=exit_time,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                shares=shares,
                pnl_dollar=pnl_dollar,
                pnl_pct=pnl_pct,
                confidence=confidence,
                was_correct=was_correct
            )
            
            trades.append(trade)
        
        # Calculate metrics
        winning_trades = [t for t in trades if t.pnl_dollar > 0]
        losing_trades = [t for t in trades if t.pnl_dollar <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        total_pnl = sum(t.pnl_dollar for t in trades)
        total_return_pct = ((current_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Sharpe ratio
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        # Max drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else 0
        
        # Avg win/loss
        avg_win = np.mean([t.pnl_dollar for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl_dollar for t in losing_trades])) if losing_trades else 0
        
        # Profit factor
        total_wins = sum(t.pnl_dollar for t in winning_trades)
        total_losses = abs(sum(t.pnl_dollar for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=trades
        )
        
        # Save to database
        self._save_backtest_result(result, "default_strategy")
        
        logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"{win_rate:.1%} win rate, {total_return_pct:+.1f}% return"
        )
        
        return result
    
    def _get_price_at_time(
        self,
        prices: list[dict],
        target_time: float
    ) -> float | None:
        """Get historical price closest to target time"""
        if not prices:
            return None
        
        # Find closest price
        closest = min(prices, key=lambda p: abs(p.get("timestamp", 0) - target_time))
        return closest.get("price")
    
    def walk_forward_analysis(
        self,
        predictions: list[dict[str, Any]],
        historical_prices: dict[str, list[dict]],
        training_window_days: int = 180,
        test_window_days: int = 30,
        step_days: int = 30
    ) -> list[BacktestResult]:
        """
        Walk-forward optimization
        
        Train on X days, test on Y days, step forward Z days, repeat
        
        Args:
            predictions: All predictions
            historical_prices: Price history
            training_window_days: Training period length
            test_window_days: Test period length
            step_days: Step size between iterations
        
        Returns:
            List of BacktestResult for each walk-forward period
        """
        logger.info(
            f"Walk-forward analysis: train={training_window_days}d, "
            f"test={test_window_days}d, step={step_days}d"
        )
        
        results = []
        
        # Get date range
        all_timestamps = [p.get("timestamp", 0) for p in predictions]
        start_timestamp = min(all_timestamps)
        end_timestamp = max(all_timestamps)
        
        current_timestamp = start_timestamp
        
        while current_timestamp + (training_window_days + test_window_days) * 86400 <= end_timestamp:
            # Define windows
            train_start = datetime.fromtimestamp(current_timestamp)
            train_end = datetime.fromtimestamp(current_timestamp + training_window_days * 86400)
            test_start = train_end
            test_end = datetime.fromtimestamp(current_timestamp + (training_window_days + test_window_days) * 86400)
            
            # Filter predictions for test window
            test_predictions = [
                p for p in predictions
                if test_start.timestamp() <= p.get("timestamp", 0) < test_end.timestamp()
            ]
            
            if not test_predictions:
                current_timestamp += step_days * 86400
                continue
            
            # Run backtest on test window
            result = self.run_backtest(
                test_predictions,
                historical_prices,
                test_start,
                test_end
            )
            
            results.append(result)
            
            # Step forward
            current_timestamp += step_days * 86400
        
        logger.info(f"Walk-forward analysis complete: {len(results)} periods tested")
        
        return results
    
    def _save_backtest_result(
        self,
        result: BacktestResult,
        strategy_name: str
    ):
        """Save backtest result to database"""
        with sqlite3.connect(str(BACKTEST_DB)) as conn:
            cursor = conn.execute(
                """
                INSERT INTO backtest_results 
                (strategy_name, start_date, end_date, total_trades, win_rate, 
                 total_return_pct, sharpe_ratio, max_drawdown, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_name,
                    result.start_date.timestamp(),
                    result.end_date.timestamp(),
                    result.total_trades,
                    result.win_rate,
                    result.total_return_pct,
                    result.sharpe_ratio,
                    result.max_drawdown,
                    time.time()
                )
            )
            
            backtest_id = cursor.lastrowid
            
            # Save trades
            for trade in result.trades:
                conn.execute(
                    """
                    INSERT INTO backtest_trades
                    (backtest_id, entry_time, exit_time, symbol, direction, 
                     entry_price, exit_price, shares, pnl_dollar, pnl_pct, 
                     confidence, was_correct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        backtest_id,
                        trade.entry_time,
                        trade.exit_time,
                        trade.symbol,
                        trade.direction,
                        trade.entry_price,
                        trade.exit_price,
                        trade.shares,
                        trade.pnl_dollar,
                        trade.pnl_pct,
                        trade.confidence,
                        1 if trade.was_correct else 0
                    )
                )
            
            conn.commit()


# Global instance
_backtest_engine = None


def get_backtest_engine(capital: float = 100000.0) -> BacktestEngine:
    """Get or create global backtest engine"""
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine(capital)
        logger.info(f"✅ Backtest engine initialized (capital: ${capital:,.0f})")
    return _backtest_engine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("📊 Testing Backtest Engine")
    print("=" * 60)
    
    engine = get_backtest_engine(capital=100000)
    
    # Generate sample predictions
    sample_predictions = [
        {
            "symbol": "AAPL",
            "direction": "BUY",
            "confidence": 0.75,
            "timestamp": time.time() - (30 * 86400)  # 30 days ago
        },
        {
            "symbol": "TSLA",
            "direction": "BUY",
            "confidence": 0.65,
            "timestamp": time.time() - (25 * 86400)
        },
    ]
    
    # Sample historical prices
    sample_prices = {
        "AAPL": [
            {"timestamp": time.time() - (30 * 86400), "price": 180.0},
            {"timestamp": time.time() - (29 * 86400), "price": 182.0},
        ],
        "TSLA": [
            {"timestamp": time.time() - (25 * 86400), "price": 250.0},
            {"timestamp": time.time() - (24 * 86400), "price": 255.0},
        ]
    }
    
    # Run backtest
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    result = engine.run_backtest(
        sample_predictions,
        sample_prices,
        start_date,
        end_date
    )
    
    print("\n📈 Backtest Results:")
    print(f"  Period: {result.start_date.date()} → {result.end_date.date()}")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate:.1%}")
    print(f"  Total Return: {result.total_return_pct:+.1f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.1f}%")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Avg Win: ${result.avg_win:.2f}")
    print(f"  Avg Loss: ${result.avg_loss:.2f}")
    
    print("\n✅ Backtest engine test complete")
