"""
Stage 4: Hedging Engine
Beyond 100% - Cross-Asset Hedging & Risk Management

Features: beta-neutral hedging, pairs trading, correlation-based hedging, dynamic hedge ratios.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


class HedgingEngine:
    """
    Advanced hedging engine for portfolio protection.

    Strategies:
    - Beta-neutral hedging (SPY hedge)
    - Correlation-based pairs trading
    - Dynamic hedge ratio calculation
    - Cross-asset hedging suggestions
    """

    def __init__(self, db_path: str = "data/hedging_engine.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        LOGGER.info(f"Hedging engine initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database for hedging strategies."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hedge_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                portfolio_symbol TEXT NOT NULL,
                hedge_symbol TEXT NOT NULL,
                hedge_type TEXT NOT NULL,

                -- Hedge details
                hedge_ratio REAL NOT NULL,
                hedge_size_pct REAL NOT NULL,
                correlation REAL,
                beta REAL,

                -- Metrics
                expected_reduction_volatility REAL,
                cost_estimate_pct REAL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pairs_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol_long TEXT NOT NULL,
                symbol_short TEXT NOT NULL,

                -- Trade details
                correlation REAL NOT NULL,
                spread REAL NOT NULL,
                z_score REAL NOT NULL,

                -- Position sizing
                long_size_pct REAL NOT NULL,
                short_size_pct REAL NOT NULL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def calculate_beta_hedge(
        self,
        portfolio_symbol: str,
        portfolio_returns: list[float],
        market_returns: list[float],  # Usually SPY
        hedge_symbol: str = "SPY",
    ) -> dict:
        """
        Calculate beta-neutral hedge ratio.

        Beta = Cov(portfolio, market) / Var(market)
        Hedge ratio = Beta (to neutralize market exposure)

        Args:
            portfolio_symbol: Main portfolio symbol
            portfolio_returns: Portfolio daily returns
            market_returns: Market index returns (SPY)
            hedge_symbol: Hedge instrument (default SPY)

        Returns:
            Dict with hedge_ratio, beta, expected_vol_reduction
        """
        if len(portfolio_returns) != len(market_returns):
            return {"error": "Returns arrays must have same length"}

        if len(portfolio_returns) < 20:
            return {"error": "Need at least 20 periods of data"}

        # Convert to numpy
        port_ret = np.array(portfolio_returns)
        mkt_ret = np.array(market_returns)

        # Calculate beta
        covariance = np.cov(port_ret, mkt_ret)[0, 1]
        market_variance = np.var(mkt_ret)
        beta = covariance / (market_variance + 1e-10)

        # Hedge ratio = beta (negative for short hedge)
        hedge_ratio = -beta

        # Calculate correlation
        correlation = np.corrcoef(port_ret, mkt_ret)[0, 1]

        # Estimate volatility reduction
        port_vol = np.std(port_ret)
        np.std(mkt_ret)

        # Hedged portfolio vol = sqrt(var_port + beta^2 * var_mkt - 2*beta*cov)
        hedged_variance = np.var(port_ret) + (beta**2 * np.var(mkt_ret)) - (2 * beta * covariance)
        hedged_vol = np.sqrt(max(0, hedged_variance))

        vol_reduction = (port_vol - hedged_vol) / port_vol

        # Record recommendation
        hedge_id = self._record_hedge_recommendation(
            portfolio_symbol=portfolio_symbol,
            hedge_symbol=hedge_symbol,
            hedge_type="beta_neutral",
            hedge_ratio=hedge_ratio,
            correlation=correlation,
            beta=beta,
            vol_reduction=vol_reduction,
        )

        return {
            "hedge_id": hedge_id,
            "portfolio_symbol": portfolio_symbol,
            "hedge_symbol": hedge_symbol,
            "hedge_type": "beta_neutral",
            "beta": round(float(beta), 4),
            "correlation": round(float(correlation), 4),
            "hedge_ratio": round(float(hedge_ratio), 4),
            "hedge_size_pct": round(abs(float(hedge_ratio)) * 100, 2),
            "expected_vol_reduction_pct": round(float(vol_reduction) * 100, 2),
            "recommendation": f"Short {abs(hedge_ratio) * 100:.0f}% of portfolio value in {hedge_symbol}",
        }

    def find_pairs_trade(
        self,
        symbol_a: str,
        returns_a: list[float],
        symbol_b: str,
        returns_b: list[float],
        entry_z_threshold: float = 2.0,
    ) -> dict:
        """
        Find pairs trading opportunity based on spread z-score.

        Args:
            symbol_a: First symbol
            returns_a: Returns for symbol A
            symbol_b: Second symbol
            returns_b: Returns for symbol B
            entry_z_threshold: Z-score threshold for entry (default 2.0)

        Returns:
            Dict with trade signal, correlation, spread, z-score
        """
        if len(returns_a) != len(returns_b):
            return {"error": "Returns arrays must have same length"}

        if len(returns_a) < 20:
            return {"error": "Need at least 20 periods"}

        # Convert to numpy
        ret_a = np.array(returns_a)
        ret_b = np.array(returns_b)

        # Calculate correlation
        correlation = np.corrcoef(ret_a, ret_b)[0, 1]

        if abs(correlation) < 0.7:
            return {
                "signal": "NO_TRADE",
                "reason": f"Correlation {correlation:.2f} too low (need |corr| > 0.7)",
                "correlation": round(float(correlation), 4),
            }

        # Calculate spread (cumulative return difference)
        cum_ret_a = np.cumsum(ret_a)
        cum_ret_b = np.cumsum(ret_b)
        spread = cum_ret_a - cum_ret_b

        # Z-score of current spread
        current_spread = spread[-1]
        mean_spread = np.mean(spread)
        std_spread = np.std(spread)
        z_score = (current_spread - mean_spread) / (std_spread + 1e-10)

        # Determine signal
        if z_score > entry_z_threshold:
            signal = "LONG_B_SHORT_A"
            recommendation = f"Long {symbol_b}, Short {symbol_a} (spread too high)"
        elif z_score < -entry_z_threshold:
            signal = "LONG_A_SHORT_B"
            recommendation = f"Long {symbol_a}, Short {symbol_b} (spread too low)"
        else:
            signal = "NO_TRADE"
            recommendation = f"Spread within normal range (z={z_score:.2f})"

        # Position sizing (equal dollar amounts)
        long_size_pct = 0.10  # 10% long
        short_size_pct = 0.10  # 10% short

        # Record if tradeable
        if signal != "NO_TRADE":
            self._record_pairs_trade(
                symbol_long=symbol_b if "B" in signal else symbol_a,
                symbol_short=symbol_a if "A" in signal else symbol_b,
                correlation=correlation,
                spread=float(current_spread),
                z_score=float(z_score),
                long_size_pct=long_size_pct,
                short_size_pct=short_size_pct,
            )

        return {
            "signal": signal,
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "correlation": round(float(correlation), 4),
            "spread": round(float(current_spread), 4),
            "z_score": round(float(z_score), 4),
            "entry_threshold": entry_z_threshold,
            "position_sizing": {
                "long_symbol": symbol_b if "B" in signal else symbol_a,
                "long_size_pct": long_size_pct * 100,
                "short_symbol": symbol_a if "A" in signal else symbol_b,
                "short_size_pct": short_size_pct * 100,
            }
            if signal != "NO_TRADE"
            else None,
            "recommendation": recommendation,
        }

    def calculate_dynamic_hedge_ratio(
        self, portfolio_returns: list[float], hedge_returns: list[float], window: int = 20
    ) -> dict:
        """
        Calculate time-varying hedge ratio using rolling window.

        Args:
            portfolio_returns: Portfolio returns
            hedge_returns: Hedge instrument returns
            window: Rolling window size (default 20)

        Returns:
            Dict with current hedge ratio and historical ratios
        """
        if len(portfolio_returns) != len(hedge_returns):
            return {"error": "Returns arrays must have same length"}

        if len(portfolio_returns) < window:
            return {"error": f"Need at least {window} periods"}

        port_ret = np.array(portfolio_returns)
        hedge_ret = np.array(hedge_returns)

        # Calculate rolling hedge ratios
        ratios = []
        for i in range(window, len(port_ret) + 1):
            window_port = port_ret[i - window : i]
            window_hedge = hedge_ret[i - window : i]

            cov = np.cov(window_port, window_hedge)[0, 1]
            var_hedge = np.var(window_hedge)
            ratio = -cov / (var_hedge + 1e-10)  # Negative for short hedge

            ratios.append(float(ratio))

        # Current hedge ratio
        current_ratio = ratios[-1] if ratios else 0.0

        # Statistics
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        return {
            "current_hedge_ratio": round(current_ratio, 4),
            "mean_hedge_ratio": round(float(mean_ratio), 4),
            "std_hedge_ratio": round(float(std_ratio), 4),
            "min_hedge_ratio": round(float(np.min(ratios)), 4),
            "max_hedge_ratio": round(float(np.max(ratios)), 4),
            "historical_ratios": [round(r, 4) for r in ratios[-10:]],  # Last 10
            "recommendation": f"Current hedge: Short {abs(current_ratio) * 100:.0f}% of portfolio",
        }

    def suggest_cross_asset_hedge(
        self,
        portfolio_returns: dict[str, list[float]],
        hedge_candidates: list[str] = None,
    ) -> list[dict]:
        """
        Suggest best cross-asset hedges for portfolio.

        Args:
            portfolio_returns: Dict of {symbol: returns} for portfolio
            hedge_candidates: List of hedge instrument symbols

        Returns:
            List of hedge suggestions ranked by effectiveness
        """
        # Calculate portfolio composite returns (equal-weighted)
        if hedge_candidates is None:
            hedge_candidates = ["SPY", "TLT", "GLD", "VXX"]
        portfolio_symbols = list(portfolio_returns.keys())
        portfolio_composite = np.mean([portfolio_returns[s] for s in portfolio_symbols], axis=0)

        hedge_suggestions = []

        for hedge_symbol in hedge_candidates:
            if hedge_symbol in portfolio_returns:
                hedge_ret = np.array(portfolio_returns[hedge_symbol])

                # Calculate correlation
                correlation = np.corrcoef(portfolio_composite, hedge_ret)[0, 1]

                # Calculate beta
                cov = np.cov(portfolio_composite, hedge_ret)[0, 1]
                var_hedge = np.var(hedge_ret)
                beta = cov / (var_hedge + 1e-10)

                # Effectiveness score (negative correlation = better hedge)
                effectiveness = -correlation if correlation < 0 else 0.0

                hedge_suggestions.append(
                    {
                        "hedge_symbol": hedge_symbol,
                        "correlation": round(float(correlation), 4),
                        "beta": round(float(beta), 4),
                        "hedge_ratio": round(float(-beta), 4),
                        "effectiveness_score": round(effectiveness, 4),
                        "hedge_quality": "Excellent"
                        if effectiveness > 0.5
                        else "Good"
                        if effectiveness > 0.3
                        else "Moderate"
                        if effectiveness > 0
                        else "Poor",
                    }
                )

        # Sort by effectiveness
        hedge_suggestions.sort(key=lambda x: x["effectiveness_score"], reverse=True)

        return hedge_suggestions

    def _record_hedge_recommendation(
        self,
        portfolio_symbol: str,
        hedge_symbol: str,
        hedge_type: str,
        hedge_ratio: float,
        correlation: float,
        beta: float,
        vol_reduction: float,
    ) -> int:
        """Record hedge recommendation."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO hedge_recommendations (
                    timestamp, portfolio_symbol, hedge_symbol, hedge_type,
                    hedge_ratio, hedge_size_pct, correlation, beta,
                    expected_reduction_volatility
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.utcnow().isoformat(),
                    portfolio_symbol,
                    hedge_symbol,
                    hedge_type,
                    hedge_ratio,
                    abs(hedge_ratio) * 100,
                    correlation,
                    beta,
                    vol_reduction * 100,
                ),
            )

            hedge_id = cursor.lastrowid
            conn.commit()
            conn.close()

            assert hedge_id is not None
            return hedge_id
        except Exception as e:
            LOGGER.error(f"Failed to record hedge: {e}")
            return -1

    def _record_pairs_trade(
        self,
        symbol_long: str,
        symbol_short: str,
        correlation: float,
        spread: float,
        z_score: float,
        long_size_pct: float,
        short_size_pct: float,
    ):
        """Record pairs trade."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO pairs_trades (
                    timestamp, symbol_long, symbol_short,
                    correlation, spread, z_score,
                    long_size_pct, short_size_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.utcnow().isoformat(),
                    symbol_long,
                    symbol_short,
                    correlation,
                    spread,
                    z_score,
                    long_size_pct,
                    short_size_pct,
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record pairs trade: {e}")


# Singleton instance
_hedging_engine: HedgingEngine | None = None


def get_hedging_engine() -> HedgingEngine:
    """Get singleton hedging engine instance."""
    global _hedging_engine
    if _hedging_engine is None:
        _hedging_engine = HedgingEngine()
    return _hedging_engine
