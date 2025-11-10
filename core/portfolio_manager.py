"""
Stage 4: Portfolio Manager
Beyond 100% - Advanced Portfolio Optimization

Multi-asset portfolio optimization using Modern Portfolio Theory (MPT).
Features: correlation analysis, Sharpe ratio maximization, efficient frontier, rebalancing.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


class PortfolioManager:
    """
    Multi-asset portfolio manager with MPT optimization.

    Features:
    - Efficient frontier calculation
    - Sharpe ratio maximization
    - Correlation matrix analysis
    - Risk parity allocation
    - Rebalancing recommendations
    """

    def __init__(self, db_path: str = "data/portfolio_manager.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Portfolio constraints
        self.min_weight = 0.05  # Min 5% per asset
        self.max_weight = 0.40  # Max 40% per asset
        self.target_sharpe = 1.5

        self._init_db()
        LOGGER.info(f"Portfolio manager initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database for portfolio allocations."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_allocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                portfolio_name TEXT NOT NULL,
                optimization_method TEXT NOT NULL,

                -- Metrics
                expected_return REAL,
                expected_volatility REAL,
                sharpe_ratio REAL,

                -- Allocations (JSON)
                weights TEXT NOT NULL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rebalance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                portfolio_name TEXT NOT NULL,

                -- Changes
                old_weights TEXT NOT NULL,
                new_weights TEXT NOT NULL,
                trades_required TEXT NOT NULL,

                -- Reason
                rebalance_reason TEXT,
                drift_threshold REAL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_portfolio_time
            ON portfolio_allocations(timestamp DESC)
        """)

        conn.commit()
        conn.close()

    def optimize_portfolio(
        self,
        assets: list[str],
        returns: dict[str, list[float]],
        target_return: float | None = None,
        risk_free_rate: float = 0.02,
    ) -> dict:
        """
        Optimize portfolio using Modern Portfolio Theory.

        Args:
            assets: List of ticker symbols
            returns: Dict of {symbol: [daily_returns]}
            target_return: Target return (if None, maximize Sharpe)
            risk_free_rate: Risk-free rate (default 2% annual)

        Returns:
            Dict with optimal weights, metrics, efficient frontier
        """
        if len(assets) < 2:
            return {"error": "Need at least 2 assets for optimization"}

        # Convert returns to numpy arrays
        returns_matrix = []
        for asset in assets:
            if asset not in returns or not returns[asset]:
                return {"error": f"No returns data for {asset}"}
            returns_matrix.append(returns[asset])

        returns_matrix = np.array(returns_matrix)

        # Calculate statistics
        mean_returns = np.mean(returns_matrix, axis=1)
        cov_matrix = np.cov(returns_matrix)

        # Annualize (assuming daily returns)
        annual_returns = mean_returns * 252
        annual_cov = cov_matrix * 252

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns_matrix)

        # Optimize
        if target_return is None:
            # Maximize Sharpe ratio
            weights, metrics = self._max_sharpe_portfolio(
                annual_returns, annual_cov, risk_free_rate
            )
            method = "max_sharpe"
        else:
            # Target return portfolio
            weights, metrics = self._target_return_portfolio(
                annual_returns, annual_cov, target_return
            )
            method = "target_return"

        # Calculate efficient frontier
        frontier = self._calculate_efficient_frontier(annual_returns, annual_cov, risk_free_rate)

        # Record allocation
        allocation_id = self._record_allocation(
            portfolio_name="optimized",
            method=method,
            weights={assets[i]: float(w) for i, w in enumerate(weights)},
            metrics=metrics,
        )

        return {
            "allocation_id": allocation_id,
            "method": method,
            "weights": {assets[i]: round(float(w), 4) for i, w in enumerate(weights)},
            "metrics": {
                "expected_return": round(metrics["return"], 4),
                "expected_volatility": round(metrics["volatility"], 4),
                "sharpe_ratio": round(metrics["sharpe"], 4),
            },
            "correlation_matrix": {
                assets[i]: {
                    assets[j]: round(float(corr_matrix[i, j]), 3) for j in range(len(assets))
                }
                for i in range(len(assets))
            },
            "efficient_frontier": frontier,
        }

    def _max_sharpe_portfolio(
        self, returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float
    ) -> tuple[np.ndarray, dict]:
        """
        Find portfolio with maximum Sharpe ratio.
        Uses simple optimization (equal weight starting point + gradient).
        """
        n_assets = len(returns)

        # Start with equal weights
        weights = np.array([1.0 / n_assets] * n_assets)

        # Iterative optimization (simple gradient ascent)
        learning_rate = 0.01
        for _ in range(1000):
            # Calculate current Sharpe
            portfolio_return = np.dot(weights, returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - risk_free_rate) / (portfolio_vol + 1e-10)

            # Gradient (approximate)
            grad = np.zeros(n_assets)
            epsilon = 0.001
            for i in range(n_assets):
                weights_plus = weights.copy()
                weights_plus[i] += epsilon
                weights_plus /= weights_plus.sum()  # Normalize

                ret_plus = np.dot(weights_plus, returns)
                vol_plus = np.sqrt(np.dot(weights_plus.T, np.dot(cov_matrix, weights_plus)))
                sharpe_plus = (ret_plus - risk_free_rate) / (vol_plus + 1e-10)

                grad[i] = (sharpe_plus - sharpe) / epsilon

            # Update weights
            weights += learning_rate * grad
            weights = np.maximum(weights, self.min_weight)
            weights = np.minimum(weights, self.max_weight)
            weights /= weights.sum()  # Normalize

        # Final metrics
        portfolio_return = np.dot(weights, returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (portfolio_return - risk_free_rate) / (portfolio_vol + 1e-10)

        metrics = {
            "return": float(portfolio_return),
            "volatility": float(portfolio_vol),
            "sharpe": float(sharpe),
        }

        return weights, metrics

    def _target_return_portfolio(
        self, returns: np.ndarray, cov_matrix: np.ndarray, target_return: float
    ) -> tuple[np.ndarray, dict]:
        """
        Find minimum variance portfolio for target return.
        """
        n_assets = len(returns)

        # Start with equal weights
        weights = np.array([1.0 / n_assets] * n_assets)

        # Iterative optimization (minimize variance, constrain return)
        learning_rate = 0.01
        for _ in range(1000):
            portfolio_return = np.dot(weights, returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            # Adjust weights to target return
            if portfolio_return < target_return:
                # Increase weight on high-return assets
                delta = returns - portfolio_return
                weights += learning_rate * delta
            else:
                # Decrease variance
                grad_var = 2 * np.dot(cov_matrix, weights)
                weights -= learning_rate * grad_var

            # Apply constraints
            weights = np.maximum(weights, self.min_weight)
            weights = np.minimum(weights, self.max_weight)
            weights /= weights.sum()

        # Final metrics
        portfolio_return = np.dot(weights, returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = portfolio_return / (portfolio_vol + 1e-10)

        metrics = {
            "return": float(portfolio_return),
            "volatility": float(portfolio_vol),
            "sharpe": float(sharpe),
        }

        return weights, metrics

    def _calculate_efficient_frontier(
        self, returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float, n_points: int = 20
    ) -> list[dict]:
        """Calculate efficient frontier points."""
        min_return = np.min(returns)
        max_return = np.max(returns)

        target_returns = np.linspace(min_return, max_return, n_points)
        frontier = []

        for target_ret in target_returns:
            weights, metrics = self._target_return_portfolio(returns, cov_matrix, target_ret)

            frontier.append(
                {
                    "return": round(float(metrics["return"]), 4),
                    "volatility": round(float(metrics["volatility"]), 4),
                    "sharpe": round(float(metrics["sharpe"]), 4),
                }
            )

        return frontier

    def calculate_risk_parity(self, assets: list[str], returns: dict[str, list[float]]) -> dict:
        """
        Risk parity allocation: equal risk contribution from each asset.
        """
        if len(assets) < 2:
            return {"error": "Need at least 2 assets"}

        # Convert returns to numpy
        returns_matrix = []
        for asset in assets:
            if asset not in returns or not returns[asset]:
                return {"error": f"No returns data for {asset}"}
            returns_matrix.append(returns[asset])

        returns_matrix = np.array(returns_matrix)

        # Calculate covariance
        cov_matrix = np.cov(returns_matrix) * 252  # Annualize

        # Risk parity: weight inversely proportional to volatility
        vols = np.sqrt(np.diag(cov_matrix))
        inverse_vols = 1.0 / vols
        weights = inverse_vols / inverse_vols.sum()

        # Calculate metrics
        mean_returns = np.mean(returns_matrix, axis=1) * 252
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = portfolio_return / (portfolio_vol + 1e-10)

        return {
            "method": "risk_parity",
            "weights": {assets[i]: round(float(w), 4) for i, w in enumerate(weights)},
            "metrics": {
                "expected_return": round(float(portfolio_return), 4),
                "expected_volatility": round(float(portfolio_vol), 4),
                "sharpe_ratio": round(float(sharpe), 4),
            },
        }

    def check_rebalance_needed(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        drift_threshold: float = 0.05,
    ) -> dict:
        """
        Check if rebalancing is needed based on drift from target.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            drift_threshold: Rebalance if any asset drifts > threshold (default 5%)

        Returns:
            Dict with rebalance_needed (bool), drifts, trades_required
        """
        drifts = {}
        max_drift = 0.0

        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        for asset in all_assets:
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            drift = current - target
            drifts[asset] = round(drift, 4)
            max_drift = max(max_drift, abs(drift))

        rebalance_needed = max_drift > drift_threshold

        # Calculate trades required
        trades = {}
        if rebalance_needed:
            for asset, drift in drifts.items():
                if abs(drift) > 0.01:  # Only trade if > 1% change
                    trades[asset] = {
                        "action": "SELL" if drift > 0 else "BUY",
                        "amount_pct": abs(drift),
                    }

        return {
            "rebalance_needed": rebalance_needed,
            "max_drift": round(max_drift, 4),
            "drift_threshold": drift_threshold,
            "drifts": drifts,
            "trades_required": trades,
            "reason": f"Max drift {max_drift * 100:.1f}% {'exceeds' if rebalance_needed else 'within'} threshold {drift_threshold * 100:.1f}%",
        }

    def _record_allocation(
        self, portfolio_name: str, method: str, weights: dict[str, float], metrics: dict
    ) -> int:
        """Record portfolio allocation in database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO portfolio_allocations (
                    timestamp, portfolio_name, optimization_method,
                    expected_return, expected_volatility, sharpe_ratio,
                    weights
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.utcnow().isoformat(),
                    portfolio_name,
                    method,
                    metrics["return"],
                    metrics["volatility"],
                    metrics["sharpe"],
                    json.dumps(weights),
                ),
            )

            allocation_id = cursor.lastrowid
            conn.commit()
            conn.close()

            assert allocation_id is not None
            return allocation_id
        except Exception as e:
            LOGGER.error(f"Failed to record allocation: {e}")
            return -1

    def get_allocation_history(
        self, portfolio_name: str = "optimized", limit: int = 10
    ) -> list[dict]:
        """Get recent allocation history."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT timestamp, optimization_method, expected_return,
                   expected_volatility, sharpe_ratio, weights
            FROM portfolio_allocations
            WHERE portfolio_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (portfolio_name, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append(
                {
                    "timestamp": row[0],
                    "method": row[1],
                    "expected_return": row[2],
                    "expected_volatility": row[3],
                    "sharpe_ratio": row[4],
                    "weights": json.loads(row[5]),
                }
            )

        return history


# Singleton instance
_portfolio_manager: PortfolioManager | None = None


def get_portfolio_manager() -> PortfolioManager:
    """Get singleton portfolio manager instance."""
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = PortfolioManager()
    return _portfolio_manager
