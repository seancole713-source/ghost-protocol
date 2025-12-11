"""
Phase 7: Advanced Analytics Engine
Calculates Sharpe ratio, drawdown, win/loss metrics, and strategy performance.
"""
import logging
import numpy as np
from datetime import datetime, timedelta, UTC
from typing import Any, Optional
from collections import defaultdict

LOGGER = logging.getLogger(__name__)


class AdvancedAnalytics:
    """Calculate advanced trading analytics and performance metrics."""
    
    def __init__(self):
        self.returns_history: list[float] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.trade_results: list[dict] = []
        self.strategy_performance: dict[str, dict] = defaultdict(lambda: {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0
        })
    
    def add_trade_result(self, pnl: float, strategy: str = "default", 
                        entry_price: float = 0.0, exit_price: float = 0.0,
                        timestamp: Optional[datetime] = None) -> None:
        """
        Record a trade result for analytics.
        
        Args:
            pnl: Profit/loss from the trade
            strategy: Strategy name (for multi-strategy comparison)
            entry_price: Entry price
            exit_price: Exit price
            timestamp: Trade timestamp
        """
        ts = timestamp or datetime.now(UTC)
        
        # Record trade
        self.trade_results.append({
            "pnl": pnl,
            "strategy": strategy,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "timestamp": ts,
            "return": (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
        })
        
        # Update returns history
        if entry_price > 0:
            ret = (exit_price - entry_price) / entry_price
            self.returns_history.append(ret)
        
        # Update equity curve
        current_equity = self.equity_curve[-1][1] if self.equity_curve else 100000.0
        new_equity = current_equity + pnl
        self.equity_curve.append((ts, new_equity))
        
        # Update strategy stats
        stats = self.strategy_performance[strategy]
        stats["trades"] += 1
        stats["total_pnl"] += pnl
        
        if pnl > 0:
            stats["wins"] += 1
            stats["avg_win"] = (stats["avg_win"] * (stats["wins"] - 1) + pnl) / stats["wins"]
        else:
            stats["losses"] += 1
            stats["avg_loss"] = (stats["avg_loss"] * (stats["losses"] - 1) + pnl) / stats["losses"]
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio (annualized).
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        
        Returns:
            Sharpe ratio
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (focuses on downside deviation).
        
        Args:
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sortino ratio
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - (risk_free_rate / 252)
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        return float(sortino)
    
    def calculate_max_drawdown(self) -> dict[str, Any]:
        """
        Calculate maximum drawdown and duration.
        
        Returns:
            Dict with max_drawdown (%), max_drawdown_duration (days), current_drawdown (%)
        """
        if len(self.equity_curve) < 2:
            return {
                "max_drawdown_pct": 0.0,
                "max_drawdown_duration_days": 0,
                "current_drawdown_pct": 0.0,
                "peak_equity": 100000.0,
                "current_equity": 100000.0
            }
        
        equity_values = [eq[1] for eq in self.equity_curve]
        timestamps = [eq[0] for eq in self.equity_curve]
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_values)
        
        # Calculate drawdowns
        drawdowns = (equity_values - running_max) / running_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        # Find max drawdown duration
        max_dd_duration = 0
        current_dd_duration = 0
        in_drawdown = False
        
        for i, dd in enumerate(drawdowns):
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    dd_start = i
                current_dd_duration = i - dd_start
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                in_drawdown = False
                current_dd_duration = 0
        
        # Convert duration to days
        if max_dd_duration > 1:
            days = (timestamps[min(dd_start + max_dd_duration, len(timestamps) - 1)] - 
                   timestamps[dd_start]).total_seconds() / 86400
        else:
            days = 0
        
        current_equity = equity_values[-1]
        peak_equity = running_max[-1]
        current_dd = (current_equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        
        return {
            "max_drawdown_pct": max_dd * 100,
            "max_drawdown_duration_days": int(days),
            "current_drawdown_pct": current_dd * 100,
            "peak_equity": float(peak_equity),
            "current_equity": float(current_equity)
        }
    
    def calculate_win_loss_metrics(self) -> dict[str, Any]:
        """
        Calculate comprehensive win/loss statistics.
        
        Returns:
            Dict with win_rate, profit_factor, avg_win, avg_loss, etc.
        """
        if not self.trade_results:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_trade_pnl": 0.0
            }
        
        wins = [t["pnl"] for t in self.trade_results if t["pnl"] > 0]
        losses = [t["pnl"] for t in self.trade_results if t["pnl"] < 0]
        
        total_trades = len(self.trade_results)
        winning_trades = len(wins)
        losing_trades = len(losses)
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "largest_win": float(max(wins)) if wins else 0.0,
            "largest_loss": float(min(losses)) if losses else 0.0,
            "avg_trade_pnl": float(np.mean([t["pnl"] for t in self.trade_results]))
        }
    
    def compare_strategies(self) -> dict[str, dict]:
        """
        Compare performance across different strategies.
        
        Returns:
            Dict mapping strategy name to performance metrics
        """
        comparison = {}
        
        for strategy, stats in self.strategy_performance.items():
            total_trades = stats["trades"]
            if total_trades == 0:
                continue
            
            win_rate = (stats["wins"] / total_trades * 100) if total_trades > 0 else 0.0
            
            total_wins = stats["avg_win"] * stats["wins"]
            total_losses = abs(stats["avg_loss"] * stats["losses"])
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            comparison[strategy] = {
                "trades": total_trades,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate_pct": win_rate,
                "total_pnl": stats["total_pnl"],
                "avg_pnl_per_trade": stats["total_pnl"] / total_trades,
                "profit_factor": profit_factor,
                "avg_win": stats["avg_win"],
                "avg_loss": stats["avg_loss"]
            }
        
        return comparison
    
    def get_comprehensive_report(self) -> dict[str, Any]:
        """
        Generate comprehensive analytics report.
        
        Returns:
            Dict with all analytics metrics
        """
        return {
            "ok": True,
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "sortino_ratio": self.calculate_sortino_ratio(),
            "drawdown": self.calculate_max_drawdown(),
            "win_loss_metrics": self.calculate_win_loss_metrics(),
            "strategy_comparison": self.compare_strategies(),
            "total_returns": float(self.returns_history[-1]) if self.returns_history else 0.0,
            "num_trades": len(self.trade_results),
            "timestamp": datetime.now(UTC).isoformat()
        }


# Global analytics instance
_analytics = AdvancedAnalytics()


def get_analytics() -> AdvancedAnalytics:
    """Get global analytics instance."""
    return _analytics


def record_trade_for_analytics(pnl: float, strategy: str = "phase5_autonomous",
                               entry_price: float = 0.0, exit_price: float = 0.0) -> None:
    """Record trade result in analytics engine."""
    _analytics.add_trade_result(pnl, strategy, entry_price, exit_price)
    LOGGER.info(f"[ANALYTICS] Recorded trade: PnL=${pnl:.2f}, Strategy={strategy}")


def get_analytics_report() -> dict[str, Any]:
    """Get comprehensive analytics report."""
    return _analytics.get_comprehensive_report()
