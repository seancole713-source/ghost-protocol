"""
Stage 3: Advanced Risk Engine
Level 9→10 (90%→100%)

Kelly criterion position sizing, VaR calculation, drawdown monitoring.
Integrates with regime detector for adaptive risk management.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


class RiskEngine:
    """
    Advanced risk management engine.

    Features:
    - Kelly criterion position sizing
    - Value at Risk (VaR) calculation
    - Maximum drawdown monitoring
    - Portfolio correlation analysis
    - Regime-adaptive risk limits
    """

    def __init__(
        self,
        db_path: str = "data/risk_metrics.db",
        max_drawdown_pct: float = 15.0,
        var_confidence: float = 0.95,
        max_single_position_pct: float = 10.0,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Risk limits
        self.max_drawdown_pct = max_drawdown_pct
        self.var_confidence = var_confidence
        self.max_single_position_pct = max_single_position_pct

        # Current portfolio state
        self.portfolio_value = 10000.0  # Default
        self.peak_value = 10000.0
        self.current_drawdown_pct = 0.0

        self._init_db()
        self._load_portfolio_state()
        LOGGER.info(f"Risk engine initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database for risk metrics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                peak_value REAL NOT NULL,
                drawdown_pct REAL NOT NULL,
                var_95 REAL,
                positions_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_risks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                position_size_usd REAL NOT NULL,
                position_pct REAL NOT NULL,
                kelly_fraction REAL,
                var_contribution REAL,
                stop_loss_pct REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_portfolio_time
            ON portfolio_snapshots(timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_position_symbol_time
            ON position_risks(symbol, timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_position_size
            ON position_risks(position_pct DESC)
        """)

        conn.commit()
        conn.close()

    def _load_portfolio_state(self):
        """Load latest portfolio state."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT portfolio_value, peak_value, drawdown_pct
                FROM portfolio_snapshots
                ORDER BY timestamp DESC
                LIMIT 1
            """)

            row = cursor.fetchone()
            if row:
                self.portfolio_value, self.peak_value, self.current_drawdown_pct = row

            conn.close()
        except Exception as e:
            LOGGER.warning(f"Could not load portfolio state: {e}")

    def update_portfolio_value(self, value: float):
        """Update current portfolio value and calculate drawdown."""
        self.portfolio_value = value

        # Update peak
        if value > self.peak_value:
            self.peak_value = value

        # Calculate drawdown
        self.current_drawdown_pct = ((self.peak_value - value) / self.peak_value) * 100

        # Record snapshot
        self._record_snapshot()

    def _record_snapshot(self):
        """Record portfolio snapshot."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO portfolio_snapshots (
                    timestamp, portfolio_value, peak_value, drawdown_pct
                ) VALUES (?, ?, ?, ?)
            """,
                (
                    datetime.utcnow().isoformat(),
                    self.portfolio_value,
                    self.peak_value,
                    self.current_drawdown_pct,
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record snapshot: {e}")

    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly criterion fraction for position sizing.

        Kelly% = (p*b - q) / b
        where:
        - p = win probability
        - q = loss probability (1-p)
        - b = win/loss ratio

        Args:
            win_rate: Probability of winning trade (0-1)
            avg_win: Average win amount (%)
            avg_loss: Average loss amount (%) - positive number

        Returns:
            Kelly fraction (0-1), capped at 0.25 (quarter Kelly)
        """
        if avg_loss == 0:
            return 0.05  # Conservative fallback

        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss

        kelly = (p * b - q) / b

        # Cap at quarter Kelly (conservative)
        kelly_capped = min(kelly, 0.25)

        # Never go negative or over 25%
        return max(0.0, min(0.25, kelly_capped))

    def calculate_var(
        self, returns: list[float], confidence: float = 0.95, portfolio_value: float | None = None
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.

        Args:
            returns: Historical returns (as percentages)
            confidence: Confidence level (default 0.95 = 95%)
            portfolio_value: Current portfolio value (default: self.portfolio_value)

        Returns:
            VaR in dollars (maximum expected loss at confidence level)
        """
        if not returns or len(returns) < 10:
            # Fallback: 5% of portfolio
            return (portfolio_value or self.portfolio_value) * 0.05

        returns_array = np.array(returns)

        # Sort returns (ascending, so worst losses are first)
        sorted_returns = np.sort(returns_array)

        # Find percentile
        percentile = 1 - confidence
        index = int(len(sorted_returns) * percentile)
        var_return_pct = sorted_returns[index]

        # Convert to dollar amount
        pv = portfolio_value or self.portfolio_value
        var_usd = abs(var_return_pct / 100.0 * pv)

        return var_usd

    def check_position_limits(
        self, symbol: str, position_size_usd: float, regime: str = "SIDEWAYS"
    ) -> dict:
        """
        Check if proposed position passes risk limits.

        Args:
            symbol: Ticker symbol
            position_size_usd: Proposed position size in USD
            regime: Current market regime

        Returns:
            Dict with approved (bool), reason, adjustments
        """
        checks = []
        approved = True
        adjusted_size = position_size_usd

        # Check 1: Single position limit (10% default)
        position_pct = (position_size_usd / self.portfolio_value) * 100
        max_pct = self.max_single_position_pct

        if position_pct > max_pct:
            checks.append(
                {
                    "check": "single_position_limit",
                    "passed": False,
                    "reason": f"Position {position_pct:.1f}% exceeds max {max_pct}%",
                    "adjusted_size": self.portfolio_value * (max_pct / 100.0),
                }
            )
            approved = False
            adjusted_size = self.portfolio_value * (max_pct / 100.0)
        else:
            checks.append(
                {
                    "check": "single_position_limit",
                    "passed": True,
                    "reason": f"Position {position_pct:.1f}% within limit",
                }
            )

        # Check 2: Drawdown limit
        if self.current_drawdown_pct >= self.max_drawdown_pct:
            checks.append(
                {
                    "check": "max_drawdown",
                    "passed": False,
                    "reason": f"Drawdown {self.current_drawdown_pct:.1f}% exceeds max {self.max_drawdown_pct}%",
                    "action": "HALT_TRADING",
                }
            )
            approved = False
            adjusted_size = 0.0
        else:
            checks.append(
                {
                    "check": "max_drawdown",
                    "passed": True,
                    "reason": f"Drawdown {self.current_drawdown_pct:.1f}% acceptable",
                }
            )

        # Check 3: Regime-adaptive sizing
        regime_multipliers = {"BULL": 1.2, "BEAR": 0.6, "SIDEWAYS": 0.8, "VOLATILE": 0.5}

        regime_mult = regime_multipliers.get(regime, 0.8)
        regime_adjusted = position_size_usd * regime_mult

        if regime_adjusted < position_size_usd:
            checks.append(
                {
                    "check": "regime_adjustment",
                    "passed": True,
                    "reason": f"Reduced to {regime_mult * 100:.0f}% due to {regime} regime",
                    "adjusted_size": regime_adjusted,
                }
            )
            adjusted_size = min(adjusted_size, regime_adjusted)
        else:
            checks.append(
                {
                    "check": "regime_adjustment",
                    "passed": True,
                    "reason": f"Increased to {regime_mult * 100:.0f}% for {regime} regime",
                    "adjusted_size": regime_adjusted,
                }
            )
            if approved:  # Only increase if not already rejected
                adjusted_size = max(adjusted_size, regime_adjusted)

        return {
            "approved": approved,
            "original_size_usd": position_size_usd,
            "adjusted_size_usd": round(adjusted_size, 2),
            "checks": checks,
            "portfolio_value": self.portfolio_value,
            "current_drawdown_pct": self.current_drawdown_pct,
        }

    def calculate_stop_loss(
        self, entry_price: float, regime: str = "SIDEWAYS", volatility: float = 0.02
    ) -> dict:
        """
        Calculate regime-adaptive stop loss.

        Args:
            entry_price: Entry price
            regime: Current market regime
            volatility: Recent volatility (as decimal)

        Returns:
            Dict with stop_loss_price, stop_loss_pct, reason
        """
        # Base stop loss percentages by regime
        base_stops = {
            "BULL": 0.08,  # Wider stops in bull
            "BEAR": 0.05,  # Tighter stops in bear
            "SIDEWAYS": 0.06,  # Moderate stops
            "VOLATILE": 0.04,  # Very tight stops
        }

        base_stop_pct = base_stops.get(regime, 0.06)

        # Adjust for volatility (wider stops if more volatile)
        vol_adjustment = min(volatility * 2, 0.03)  # Cap at 3%

        final_stop_pct = base_stop_pct + vol_adjustment
        stop_loss_price = entry_price * (1 - final_stop_pct)

        return {
            "stop_loss_price": round(stop_loss_price, 2),
            "stop_loss_pct": round(final_stop_pct * 100, 2),
            "base_stop_pct": round(base_stop_pct * 100, 2),
            "volatility_adjustment_pct": round(vol_adjustment * 100, 2),
            "regime": regime,
        }

    def get_risk_dashboard(self) -> dict:
        """Get comprehensive risk metrics dashboard."""
        # Portfolio metrics
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get recent returns for VaR
        cursor.execute("""
            SELECT portfolio_value
            FROM portfolio_snapshots
            ORDER BY timestamp DESC
            LIMIT 30
        """)

        values = [row[0] for row in cursor.fetchall()]

        returns = []
        if len(values) >= 2:
            for i in range(len(values) - 1):
                ret = ((values[i] - values[i + 1]) / values[i + 1]) * 100
                returns.append(ret)

        # Calculate VaR
        var_95 = self.calculate_var(returns, 0.95) if returns else 0.0
        var_99 = self.calculate_var(returns, 0.99) if returns else 0.0

        conn.close()

        return {
            "portfolio": {
                "current_value": round(self.portfolio_value, 2),
                "peak_value": round(self.peak_value, 2),
                "current_drawdown_pct": round(self.current_drawdown_pct, 2),
                "max_drawdown_limit_pct": self.max_drawdown_pct,
            },
            "value_at_risk": {
                "var_95_usd": round(var_95, 2),
                "var_95_pct": round((var_95 / self.portfolio_value) * 100, 2),
                "var_99_usd": round(var_99, 2),
                "var_99_pct": round((var_99 / self.portfolio_value) * 100, 2),
                "confidence": self.var_confidence,
            },
            "position_limits": {
                "max_single_position_pct": self.max_single_position_pct,
                "max_single_position_usd": round(
                    self.portfolio_value * (self.max_single_position_pct / 100), 2
                ),
            },
            "status": self._get_risk_status(),
        }

    def _get_risk_status(self) -> dict:
        """Get overall risk status (green/yellow/red)."""
        if self.current_drawdown_pct >= self.max_drawdown_pct:
            return {
                "level": "red",
                "message": "HALT TRADING - Max drawdown reached",
                "action_required": True,
            }
        elif self.current_drawdown_pct >= self.max_drawdown_pct * 0.75:
            return {
                "level": "yellow",
                "message": "WARNING - Approaching max drawdown",
                "action_required": False,
            }
        else:
            return {"level": "green", "message": "Risk levels normal", "action_required": False}

    def risk_check_order(
        self,
        order: dict,
        portfolio_value: float,
        current_nav: float,
        existing_positions: dict,
    ) -> tuple[bool, str]:
        """
        Comprehensive risk check for an order before submission.
        Required by broker integration.

        Args:
            order: Order dict with {symbol, qty, price, side, type}
            portfolio_value: Total portfolio value
            current_nav: Current NAV
            existing_positions: Dict of existing positions

        Returns:
            (allowed, reason) - allowed=False if order blocked
        """
        import os

        # Check kill switch
        kill_switch = os.getenv("RISK_KILL", "0") == "1"
        if kill_switch:
            return False, "🛑 KILL SWITCH ACTIVE - All trading halted"

        # Check daily drawdown
        self.update_portfolio_value(current_nav)
        if self.current_drawdown_pct >= self.max_drawdown_pct:
            return False, (
                f"❌ Max drawdown exceeded: {self.current_drawdown_pct:.2f}% > {self.max_drawdown_pct}% "
                f"(NAV: ${current_nav:,.2f}, Peak: ${self.peak_value:,.2f})"
            )

        # Check position size (only for BUY orders)
        if order.get("side", "").lower() == "buy":
            symbol = order.get("symbol", "")
            qty = order.get("qty", 0)
            price = order.get("price", 0)

            if portfolio_value <= 0:
                return False, "❌ Portfolio value is zero or negative"

            # Calculate position value
            order_value = abs(qty * price)

            # Check if adding to existing position
            existing_value = 0
            if symbol in existing_positions:
                existing_pos = existing_positions[symbol]
                existing_value = abs(existing_pos.get("qty", 0) * existing_pos.get("price", price))

            # Total position value after order
            total_position_value = existing_value + order_value
            position_pct = (total_position_value / portfolio_value) * 100

            if position_pct > self.max_single_position_pct:
                return False, (
                    f"❌ Position size limit exceeded: {position_pct:.1f}% > {self.max_single_position_pct}% max "
                    f"(${total_position_value:,.2f} of ${portfolio_value:,.2f} portfolio)"
                )

        return True, "✅ All risk checks passed"

    def scan_positions_for_exits(
        self,
        positions: list[dict],
    ) -> list[dict]:
        """
        Scan all positions and identify which should be exited due to SL/TP.
        Required by broker integration.

        Args:
            positions: List of positions [{symbol, qty, avg_cost, current_price}]

        Returns:
            List of exit signals [{symbol, reason, type: 'stop_loss' | 'take_profit'}]
        """
        import os

        # Get SL/TP thresholds from environment
        sl_pct = float(os.getenv("RISK_SL_PCT", "3"))
        tp_pct = float(os.getenv("RISK_TP_PCT", "6"))

        exit_signals = []

        for pos in positions:
            symbol = pos.get("symbol", "")
            avg_cost = pos.get("avg_cost", 0) or pos.get("entry_price", 0)
            current_price = pos.get("current_price", 0) or pos.get("price", 0)

            if not symbol or not avg_cost or not current_price:
                continue

            # Calculate P&L percentage
            pnl_pct = ((current_price - avg_cost) / avg_cost) * 100

            # Check stop-loss
            if pnl_pct <= -sl_pct:
                exit_signals.append(
                    {
                        "symbol": symbol,
                        "type": "stop_loss",
                        "reason": (
                            f"🛑 STOP-LOSS triggered for {symbol}: {pnl_pct:.2f}% loss "
                            f"(limit: {sl_pct}%) - Price ${current_price:.2f} vs Entry ${avg_cost:.2f}"
                        ),
                        "pnl_pct": pnl_pct,
                        "entry_price": avg_cost,
                        "current_price": current_price,
                    }
                )
                continue  # Don't check TP if SL triggered

            # Check take-profit
            if pnl_pct >= tp_pct:
                exit_signals.append(
                    {
                        "symbol": symbol,
                        "type": "take_profit",
                        "reason": (
                            f"💰 TAKE-PROFIT triggered for {symbol}: {pnl_pct:.2f}% gain "
                            f"(target: {tp_pct}%) - Price ${current_price:.2f} vs Entry ${avg_cost:.2f}"
                        ),
                        "pnl_pct": pnl_pct,
                        "entry_price": avg_cost,
                        "current_price": current_price,
                    }
                )

        return exit_signals

    def get_status(self) -> dict:
        """
        Get current risk engine status for API.
        Required by broker integration.
        """
        import os

        return {
            "enabled": True,
            "kill_switch": os.getenv("RISK_KILL", "0") == "1",
            "limits": {
                "max_position_pct": self.max_single_position_pct,
                "stop_loss_pct": float(os.getenv("RISK_SL_PCT", "3")),
                "take_profit_pct": float(os.getenv("RISK_TP_PCT", "6")),
                "max_daily_drawdown_pct": self.max_drawdown_pct,
            },
            "current": {
                "portfolio_value": self.portfolio_value,
                "peak_value": self.peak_value,
                "drawdown_pct": self.current_drawdown_pct,
            },
            "status": self._get_risk_status(),
        }


# Singleton instance
_risk_engine: RiskEngine | None = None


def get_risk_engine() -> RiskEngine:
    """Get singleton risk engine instance."""
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskEngine()
    return _risk_engine
