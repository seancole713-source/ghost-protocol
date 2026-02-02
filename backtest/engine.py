"""
Backtest Engine
Core backtesting logic with NO LOOKAHEAD BIAS
"""

import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Trade:
    """Record of a single trade"""
    entry_time: datetime
    exit_time: datetime
    prediction: str  # 'UP' or 'DOWN'
    entry_price: float
    exit_price: float
    pct_change: float
    outcome: str  # 'WIN' or 'LOSS'
    

class BacktestEngine:
    """
    Backtest engine that runs strategies without lookahead bias.
    
    CRITICAL: Strategy function only sees data BEFORE the prediction point.
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        strategy_fn: Callable[[pd.DataFrame], str],
        holding_hours: int = 48,
        min_lookback_hours: int = 168,  # Need at least 7 days of history
        step_hours: int = 24,  # Make prediction every 24 hours
        win_threshold: float = 0.0,  # 0% = any move in right direction is a win
    ):
        """
        Initialize backtest engine.
        
        Args:
            data: DataFrame with OHLCV columns (must have DatetimeIndex)
            strategy_fn: Function that takes historical DataFrame, returns 'UP', 'DOWN', or 'FLAT'
            holding_hours: How long to hold position before checking outcome
            min_lookback_hours: Minimum history needed for strategy
            step_hours: How often to make predictions
            win_threshold: Minimum % move to count as win (0 = any positive move)
        """
        self.data = data.copy()
        self.strategy_fn = strategy_fn
        self.holding_hours = holding_hours
        self.min_lookback_hours = min_lookback_hours
        self.step_hours = step_hours
        self.win_threshold = win_threshold
        
        # Validate data
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data missing required column: {col}")
    
    def run(self) -> Dict:
        """
        Run backtest.
        
        Returns:
            {
                'total_trades': int,
                'wins': int,
                'losses': int,
                'flats_skipped': int,
                'win_rate': float,
                'avg_win_pct': float,
                'avg_loss_pct': float,
                'trades': List[Trade]
            }
        """
        trades: List[Trade] = []
        flats_skipped = 0
        
        # Calculate indices to make predictions at
        total_bars = len(self.data)
        
        # Start after min_lookback, end with room for holding period
        start_idx = self.min_lookback_hours
        end_idx = total_bars - self.holding_hours
        
        if start_idx >= end_idx:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'flats_skipped': 0,
                'win_rate': 0.0,
                'avg_win_pct': 0.0,
                'avg_loss_pct': 0.0,
                'trades': []
            }
        
        # Make predictions at regular intervals
        for idx in range(start_idx, end_idx, self.step_hours):
            # Get data available UP TO this point (NO LOOKAHEAD)
            historical_data = self.data.iloc[:idx].copy()
            
            # Get prediction from strategy
            try:
                prediction = self.strategy_fn(historical_data)
            except Exception as e:
                # Strategy error - skip this point
                continue
            
            # Skip FLAT predictions
            if prediction == 'FLAT' or prediction not in ('UP', 'DOWN'):
                flats_skipped += 1
                continue
            
            # Get entry and exit prices
            entry_time = self.data.index[idx]
            exit_idx = idx + self.holding_hours
            exit_time = self.data.index[exit_idx]
            
            entry_price = self.data['Close'].iloc[idx]
            exit_price = self.data['Close'].iloc[exit_idx]
            
            # Calculate actual move
            pct_change = (exit_price - entry_price) / entry_price
            
            # Determine outcome
            if prediction == 'UP':
                is_win = pct_change > self.win_threshold
            else:  # DOWN
                is_win = pct_change < -self.win_threshold
            
            outcome = 'WIN' if is_win else 'LOSS'
            
            trade = Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                prediction=prediction,
                entry_price=entry_price,
                exit_price=exit_price,
                pct_change=pct_change,
                outcome=outcome
            )
            trades.append(trade)
        
        # Calculate statistics
        wins = sum(1 for t in trades if t.outcome == 'WIN')
        losses = len(trades) - wins
        win_rate = wins / len(trades) if trades else 0.0
        
        winning_trades = [t for t in trades if t.outcome == 'WIN']
        losing_trades = [t for t in trades if t.outcome == 'LOSS']
        
        avg_win_pct = np.mean([abs(t.pct_change) for t in winning_trades]) if winning_trades else 0.0
        avg_loss_pct = np.mean([abs(t.pct_change) for t in losing_trades]) if losing_trades else 0.0
        
        return {
            'total_trades': len(trades),
            'wins': wins,
            'losses': losses,
            'flats_skipped': flats_skipped,
            'win_rate': win_rate,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'trades': trades
        }
    
    def run_with_details(self) -> pd.DataFrame:
        """Run backtest and return detailed trade log as DataFrame"""
        result = self.run()
        
        if not result['trades']:
            return pd.DataFrame()
        
        trade_data = []
        for t in result['trades']:
            trade_data.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'prediction': t.prediction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pct_change': t.pct_change,
                'outcome': t.outcome
            })
        
        return pd.DataFrame(trade_data)


def calculate_significance(wins: int, total: int, baseline: float = 0.50) -> Dict:
    """
    Test if win rate is statistically significant.
    
    Uses binomial test to determine if win rate > baseline.
    
    Args:
        wins: Number of winning trades
        total: Total number of trades
        baseline: Expected win rate under null hypothesis (default 50%)
        
    Returns:
        {
            'is_significant': bool (p < 0.05),
            'p_value': float,
            'confidence_interval': (lower, upper),
            'min_trades_needed': int (for significance at current rate)
        }
    """
    from scipy import stats
    
    if total == 0:
        return {
            'is_significant': False,
            'p_value': 1.0,
            'confidence_interval': (0.0, 1.0),
            'min_trades_needed': 0
        }
    
    observed_rate = wins / total
    
    # Binomial test (one-sided: is win rate > baseline?)
    # Using exact binomial test
    p_value = stats.binom_test(wins, total, baseline, alternative='greater')
    
    # 95% confidence interval for win rate (Wilson score interval)
    # More accurate for small samples than normal approximation
    z = 1.96  # 95% CI
    n = total
    p_hat = observed_rate
    
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denominator
    
    ci_lower = max(0, center - spread)
    ci_upper = min(1, center + spread)
    
    # Estimate minimum trades needed for significance
    # If current rate holds, how many trades to get p < 0.05?
    min_trades_needed = 0
    if observed_rate > baseline:
        for n_trades in range(total, 1000):
            expected_wins = int(n_trades * observed_rate)
            test_p = stats.binom_test(expected_wins, n_trades, baseline, alternative='greater')
            if test_p < 0.05:
                min_trades_needed = n_trades
                break
        else:
            min_trades_needed = 1000  # Would need 1000+ trades
    
    return {
        'is_significant': p_value < 0.05,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'min_trades_needed': min_trades_needed
    }


if __name__ == "__main__":
    # Test the engine with dummy data
    print("BacktestEngine module loaded successfully")
    print("Run 'python backtest/run_backtest.py' to execute backtests")
