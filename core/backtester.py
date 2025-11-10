"""
Stage 4: Advanced Backtester
Beyond 100% - Monte Carlo & Walk-Forward Analysis

Features: historical backtesting, Monte Carlo simulation, walk-forward optimization, performance attribution.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


class Backtester:
    """
    Advanced backtesting engine with Monte Carlo and walk-forward analysis.

    Features:
    - Historical strategy backtesting
    - Monte Carlo simulation (bootstrap)
    - Walk-forward optimization
    - Performance attribution
    - Drawdown analysis
    """

    def __init__(self, db_path: str = "data/backtester.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        LOGGER.info(f"Backtester initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database for backtest results."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                strategy_name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,

                -- Performance
                total_return REAL NOT NULL,
                annualized_return REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                max_drawdown REAL NOT NULL,

                -- Risk metrics
                volatility REAL NOT NULL,
                win_rate REAL,
                profit_factor REAL,

                -- Trade stats
                num_trades INTEGER,
                avg_trade_return REAL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS monte_carlo_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                simulation_id INTEGER NOT NULL,

                -- Simulated performance
                total_return REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                max_drawdown REAL NOT NULL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def run_backtest(
        self,
        strategy_name: str,
        returns: list[float],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
    ) -> dict:
        """
        Run historical backtest on strategy returns.

        Args:
            strategy_name: Name of strategy
            returns: Daily returns (decimal, e.g. 0.01 = 1%)
            start_date: ISO date string
            end_date: ISO date string
            initial_capital: Starting capital

        Returns:
            Dict with performance metrics
        """
        if len(returns) < 20:
            return {"error": "Need at least 20 periods"}

        ret_array = np.array(returns)

        # Calculate equity curve
        equity_curve = [initial_capital]
        for r in ret_array:
            equity_curve.append(equity_curve[-1] * (1 + r))

        # Total return
        total_return = (equity_curve[-1] - initial_capital) / initial_capital

        # Annualized return (assume 252 trading days)
        num_years = len(returns) / 252.0
        annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0.0

        # Volatility (annualized)
        volatility = np.std(ret_array) * np.sqrt(252)

        # Sharpe ratio (assume 0% risk-free rate)
        sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0.0

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        # Win rate
        winning_trades = sum(1 for r in ret_array if r > 0)
        win_rate = winning_trades / len(ret_array)

        # Profit factor
        gross_profit = sum(r for r in ret_array if r > 0)
        gross_loss = abs(sum(r for r in ret_array if r < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # Average trade return
        avg_trade_return = float(np.mean(ret_array))

        # Generate run ID
        run_id = f"{strategy_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Record results
        self._record_backtest(
            run_id=run_id,
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(returns),
            avg_trade_return=avg_trade_return,
        )

        return {
            "run_id": run_id,
            "strategy_name": strategy_name,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "final_equity": round(equity_curve[-1], 2),
            "performance": {
                "total_return_pct": round(total_return * 100, 2),
                "annualized_return_pct": round(annualized_return * 100, 2),
                "volatility_pct": round(volatility * 100, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "max_drawdown_pct": round(max_drawdown * 100, 2),
            },
            "trade_stats": {
                "num_trades": len(returns),
                "win_rate_pct": round(win_rate * 100, 2),
                "avg_trade_return_pct": round(avg_trade_return * 100, 4),
                "profit_factor": round(profit_factor, 2)
                if profit_factor != float("inf")
                else "inf",
            },
            "equity_curve": [round(e, 2) for e in equity_curve[-10:]],  # Last 10 points
        }

    def monte_carlo_simulation(
        self, returns: list[float], num_simulations: int = 1000, simulation_length: int = 252
    ) -> dict:
        """
        Run Monte Carlo simulation using bootstrap method.

        Args:
            returns: Historical returns
            num_simulations: Number of simulations to run
            simulation_length: Length of each simulation (default 252 = 1 year)

        Returns:
            Dict with simulation statistics
        """
        if len(returns) < 20:
            return {"error": "Need at least 20 historical returns"}

        ret_array = np.array(returns)
        run_id = f"mc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        simulation_results = []

        for sim_id in range(num_simulations):
            # Bootstrap sample
            sampled_returns = np.random.choice(ret_array, size=simulation_length, replace=True)

            # Calculate equity curve
            equity = 1.0
            equity_curve = [equity]
            for r in sampled_returns:
                equity *= 1 + r
                equity_curve.append(equity)

            # Metrics
            total_return = equity - 1.0
            vol = np.std(sampled_returns) * np.sqrt(252)
            sharpe = (total_return / vol) if vol > 0 else 0.0
            max_dd = self._calculate_max_drawdown(equity_curve)

            simulation_results.append(
                {
                    "total_return": float(total_return),
                    "sharpe_ratio": float(sharpe),
                    "max_drawdown": float(max_dd),
                }
            )

            # Record to database
            self._record_monte_carlo(run_id, sim_id, total_return, sharpe, max_dd)

        # Calculate statistics
        returns_dist = [s["total_return"] for s in simulation_results]
        sharpe_dist = [s["sharpe_ratio"] for s in simulation_results]
        dd_dist = [s["max_drawdown"] for s in simulation_results]

        # Percentiles
        return_5th = np.percentile(returns_dist, 5)
        return_50th = np.percentile(returns_dist, 50)
        return_95th = np.percentile(returns_dist, 95)

        sharpe_5th = np.percentile(sharpe_dist, 5)
        sharpe_50th = np.percentile(sharpe_dist, 50)
        sharpe_95th = np.percentile(sharpe_dist, 95)

        dd_5th = np.percentile(dd_dist, 5)
        dd_50th = np.percentile(dd_dist, 50)
        dd_95th = np.percentile(dd_dist, 95)

        return {
            "run_id": run_id,
            "num_simulations": num_simulations,
            "simulation_length": simulation_length,
            "total_return": {
                "5th_percentile_pct": round(return_5th * 100, 2),
                "median_pct": round(return_50th * 100, 2),
                "95th_percentile_pct": round(return_95th * 100, 2),
                "mean_pct": round(np.mean(returns_dist) * 100, 2),
                "std_pct": round(np.std(returns_dist) * 100, 2),
            },
            "sharpe_ratio": {
                "5th_percentile": round(sharpe_5th, 2),
                "median": round(sharpe_50th, 2),
                "95th_percentile": round(sharpe_95th, 2),
                "mean": round(np.mean(sharpe_dist), 2),
            },
            "max_drawdown": {
                "5th_percentile_pct": round(dd_5th * 100, 2),
                "median_pct": round(dd_50th * 100, 2),
                "95th_percentile_pct": round(dd_95th * 100, 2),
            },
        }

    def walk_forward_analysis(
        self,
        returns: list[float],
        in_sample_window: int = 120,
        out_sample_window: int = 30,
        step_size: int = 30,
    ) -> dict:
        """
        Walk-forward optimization analysis.

        Args:
            returns: Historical returns
            in_sample_window: Training window size (default 120 days)
            out_sample_window: Testing window size (default 30 days)
            step_size: Step between windows (default 30 days)

        Returns:
            Dict with walk-forward results
        """
        if len(returns) < (in_sample_window + out_sample_window):
            return {"error": f"Need at least {in_sample_window + out_sample_window} periods"}

        ret_array = np.array(returns)

        results = []
        current_pos = 0

        while current_pos + in_sample_window + out_sample_window <= len(ret_array):
            # In-sample data
            in_sample_start = current_pos
            in_sample_end = current_pos + in_sample_window
            in_sample_data = ret_array[in_sample_start:in_sample_end]

            # Out-of-sample data
            out_sample_start = in_sample_end
            out_sample_end = in_sample_end + out_sample_window
            out_sample_data = ret_array[out_sample_start:out_sample_end]

            # "Optimize" on in-sample (simple average as proxy)
            np.mean(in_sample_data)
            in_sample_sharpe = (np.mean(in_sample_data) * np.sqrt(252)) / (
                np.std(in_sample_data) * np.sqrt(252) + 1e-10
            )

            # Test on out-of-sample
            out_sample_return = np.sum(out_sample_data)
            out_sample_sharpe = (np.mean(out_sample_data) * np.sqrt(252)) / (
                np.std(out_sample_data) * np.sqrt(252) + 1e-10
            )

            results.append(
                {
                    "window_id": len(results) + 1,
                    "in_sample_sharpe": round(float(in_sample_sharpe), 2),
                    "out_sample_return_pct": round(float(out_sample_return) * 100, 2),
                    "out_sample_sharpe": round(float(out_sample_sharpe), 2),
                }
            )

            current_pos += step_size

        # Summary statistics
        out_sample_returns = [r["out_sample_return_pct"] for r in results]
        out_sample_sharpes = [r["out_sample_sharpe"] for r in results]

        return {
            "num_windows": len(results),
            "in_sample_window": in_sample_window,
            "out_sample_window": out_sample_window,
            "step_size": step_size,
            "summary": {
                "avg_out_sample_return_pct": round(np.mean(out_sample_returns), 2),
                "avg_out_sample_sharpe": round(np.mean(out_sample_sharpes), 2),
                "consistency": round(
                    sum(1 for r in out_sample_returns if r > 0) / len(out_sample_returns) * 100, 2
                ),
            },
            "windows": results[:5],  # First 5 windows
        }

    def performance_attribution(
        self, portfolio_returns: list[float], benchmark_returns: list[float]
    ) -> dict:
        """
        Attribute performance vs benchmark.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns (e.g. SPY)

        Returns:
            Dict with attribution metrics
        """
        if len(portfolio_returns) != len(benchmark_returns):
            return {"error": "Returns arrays must have same length"}

        port_ret = np.array(portfolio_returns)
        bench_ret = np.array(benchmark_returns)

        # Total returns
        port_total = np.prod(1 + port_ret) - 1
        bench_total = np.prod(1 + bench_ret) - 1

        # Alpha (excess return)
        alpha = port_total - bench_total

        # Beta
        cov = np.cov(port_ret, bench_ret)[0, 1]
        bench_var = np.var(bench_ret)
        beta = cov / (bench_var + 1e-10)

        # Information ratio
        active_returns = port_ret - bench_ret
        tracking_error = np.std(active_returns) * np.sqrt(252)
        information_ratio = (np.mean(active_returns) * 252) / (tracking_error + 1e-10)

        # Up/down capture
        up_periods = bench_ret > 0
        down_periods = bench_ret < 0

        up_capture = (
            (np.mean(port_ret[up_periods]) / np.mean(bench_ret[up_periods]))
            if up_periods.sum() > 0
            else 1.0
        )
        down_capture = (
            (np.mean(port_ret[down_periods]) / np.mean(bench_ret[down_periods]))
            if down_periods.sum() > 0
            else 1.0
        )

        return {
            "portfolio_return_pct": round(port_total * 100, 2),
            "benchmark_return_pct": round(bench_total * 100, 2),
            "alpha_pct": round(alpha * 100, 2),
            "beta": round(float(beta), 2),
            "information_ratio": round(float(information_ratio), 2),
            "tracking_error_pct": round(tracking_error * 100, 2),
            "up_capture_ratio": round(float(up_capture), 2),
            "down_capture_ratio": round(float(down_capture), 2),
        }

    def _calculate_max_drawdown(self, equity_curve: list[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _record_backtest(
        self,
        run_id: str,
        strategy_name: str,
        start_date: str,
        end_date: str,
        total_return: float,
        annualized_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        volatility: float,
        win_rate: float,
        profit_factor: float,
        num_trades: int,
        avg_trade_return: float,
    ):
        """Record backtest results."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO backtest_runs (
                    run_id, strategy_name, start_date, end_date,
                    total_return, annualized_return, sharpe_ratio, max_drawdown,
                    volatility, win_rate, profit_factor, num_trades, avg_trade_return
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    strategy_name,
                    start_date,
                    end_date,
                    total_return,
                    annualized_return,
                    sharpe_ratio,
                    max_drawdown,
                    volatility,
                    win_rate,
                    profit_factor,
                    num_trades,
                    avg_trade_return,
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record backtest: {e}")

    def _record_monte_carlo(
        self,
        run_id: str,
        simulation_id: int,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
    ):
        """Record Monte Carlo simulation."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO monte_carlo_runs (
                    run_id, simulation_id, total_return, sharpe_ratio, max_drawdown
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (run_id, simulation_id, total_return, sharpe_ratio, max_drawdown),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record Monte Carlo: {e}")


# Singleton instance
_backtester: Backtester | None = None


def get_backtester() -> Backtester:
    """Get singleton backtester instance."""
    global _backtester
    if _backtester is None:
        _backtester = Backtester()
    return _backtester
