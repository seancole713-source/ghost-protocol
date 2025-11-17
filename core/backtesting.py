"""
Backtesting engine for Ghost AI predictions.
Tests strategy performance on historical data.
"""

import time
from typing import Any
import yfinance as yf
import pandas as pd


def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 10000.0,
    position_size_pct: float = 0.10,
    strategy: str = "ghost-momentum"
) -> dict[str, Any]:
    """
    Run backtest on historical data.
    
    Args:
        symbol: Stock/crypto ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        position_size_pct: % of capital per trade (0.10 = 10%)
        strategy: Trading strategy ("ghost-momentum", "ghost-trend", etc.)
    
    Returns:
        {
            "ok": True/False,
            "symbol": symbol,
            "period": {start, end},
            "trades": [...],
            "performance": {
                "total_return": %,
                "total_return_dollars": $,
                "win_rate": %,
                "sharpe_ratio": N,
                "max_drawdown": %,
                "total_trades": N,
                "winners": N,
                "losers": N
            }
        }
    """
    try:
        # Fetch historical data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            return {
                "ok": False,
                "error": "No historical data available",
                "symbol": symbol
            }
        
        # Run strategy simulation
        trades = []
        capital = initial_capital
        position = None
        equity_curve = [initial_capital]
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            prev_price = df['Close'].iloc[i-1]
            
            # Simple momentum strategy: Buy if price increased, sell if decreased
            if strategy == "ghost-momentum":
                signal = _momentum_signal(df, i)
            elif strategy == "ghost-trend":
                signal = _trend_signal(df, i)
            else:
                signal = "HOLD"
            
            # Execute trades based on signal
            if signal == "BUY" and position is None:
                # Open long position
                position_size = (capital * position_size_pct) / current_price
                position = {
                    "entry_price": current_price,
                    "quantity": position_size,
                    "entry_date": df.index[i]
                }
                capital -= position_size * current_price
                
            elif signal == "SELL" and position is not None:
                # Close position
                exit_price = current_price
                pnl = (exit_price - position["entry_price"]) * position["quantity"]
                capital += position["quantity"] * exit_price
                
                trades.append({
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "quantity": position["quantity"],
                    "pnl": pnl,
                    "pnl_pct": ((exit_price - position["entry_price"]) / position["entry_price"]) * 100,
                    "entry_date": str(position["entry_date"]),
                    "exit_date": str(df.index[i]),
                    "hold_days": (df.index[i] - position["entry_date"]).days
                })
                
                position = None
            
            # Track equity
            current_equity = capital
            if position:
                current_equity += position["quantity"] * current_price
            equity_curve.append(current_equity)
        
        # Calculate performance metrics
        performance = _calculate_performance(trades, initial_capital, equity_curve[-1], equity_curve)
        
        return {
            "ok": True,
            "symbol": symbol,
            "period": {
                "start": start_date,
                "end": end_date,
                "days": len(df)
            },
            "trades": trades,
            "performance": performance,
            "final_capital": round(equity_curve[-1], 2),
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "symbol": symbol
        }


def _momentum_signal(df: pd.DataFrame, index: int) -> str:
    """Generate momentum-based signal."""
    if index < 5:
        return "HOLD"
    
    # Simple momentum: Buy if price > 5-day average, sell if below
    current_price = df['Close'].iloc[index]
    ma5 = df['Close'].iloc[index-5:index].mean()
    
    if current_price > ma5 * 1.02:  # 2% above MA
        return "BUY"
    elif current_price < ma5 * 0.98:  # 2% below MA
        return "SELL"
    
    return "HOLD"


def _trend_signal(df: pd.DataFrame, index: int) -> str:
    """Generate trend-following signal."""
    if index < 20:
        return "HOLD"
    
    # Trend: Buy if price > 20-day MA and rising, sell if below and falling
    current_price = df['Close'].iloc[index]
    ma20 = df['Close'].iloc[index-20:index].mean()
    prev_ma20 = df['Close'].iloc[index-21:index-1].mean()
    
    if current_price > ma20 and ma20 > prev_ma20:
        return "BUY"
    elif current_price < ma20 and ma20 < prev_ma20:
        return "SELL"
    
    return "HOLD"


def _calculate_performance(
    trades: list[dict[str, Any]],
    initial_capital: float,
    final_capital: float,
    equity_curve: list[float]
) -> dict[str, Any]:
    """Calculate performance metrics."""
    if not trades:
        return {
            "total_return": 0.0,
            "total_return_dollars": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "winners": 0,
            "losers": 0
        }
    
    # Total return
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    total_return_dollars = final_capital - initial_capital
    
    # Win rate
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] < 0]
    win_rate = (len(winners) / len(trades) * 100) if trades else 0.0
    
    # Max drawdown
    max_drawdown = _calculate_max_drawdown(equity_curve)
    
    # Sharpe ratio (simplified - using daily returns)
    sharpe_ratio = _calculate_sharpe_ratio(equity_curve)
    
    # Average trade metrics
    avg_winner = sum(t["pnl"] for t in winners) / len(winners) if winners else 0
    avg_loser = sum(t["pnl"] for t in losers) / len(losers) if losers else 0
    
    return {
        "total_return": round(total_return_pct, 2),
        "total_return_dollars": round(total_return_dollars, 2),
        "win_rate": round(win_rate, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown": round(max_drawdown, 2),
        "total_trades": len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "average_winner": round(avg_winner, 2),
        "average_loser": round(avg_loser, 2),
        "profit_factor": round(abs(avg_winner / avg_loser), 2) if avg_loser != 0 else 0
    }


def _calculate_max_drawdown(equity_curve: list[float]) -> float:
    """Calculate maximum drawdown %."""
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = ((peak - value) / peak) * 100
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


def _calculate_sharpe_ratio(equity_curve: list[float]) -> float:
    """Calculate Sharpe ratio (simplified)."""
    if len(equity_curve) < 2:
        return 0.0
    
    returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] for i in range(1, len(equity_curve))]
    
    if not returns:
        return 0.0
    
    avg_return = sum(returns) / len(returns)
    std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
    
    if std_return == 0:
        return 0.0
    
    # Annualized Sharpe (assuming 252 trading days)
    sharpe = (avg_return / std_return) * (252 ** 0.5)
    
    return sharpe
