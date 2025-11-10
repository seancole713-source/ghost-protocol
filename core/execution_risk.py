"""
Stage 5: Execution Risk Controls
Pre-Trade Risk Checks & Kill Switch

Features: pre-trade validation, position limits, order size checks, emergency kill switch.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class ExecutionRisk:
    """
    Pre-trade risk controls and emergency controls.

    Features:
    - Pre-trade risk checks
    - Position limit enforcement
    - Order size validation
    - Kill switch (emergency halt)
    - Risk breach logging
    """

    def __init__(self, db_path: str = "data/execution_risk.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Risk limits
        self.max_order_value = 1000000.0  # $1M per order
        self.max_position_size = 5000000.0  # $5M per position
        self.max_daily_trades = 1000
        self.max_order_quantity = 100000  # shares

        # Kill switch
        self.trading_enabled = True
        self.kill_switch_reason = None

        self._init_db()
        LOGGER.info(f"Execution risk controls initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database for risk controls."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_checks (
                check_id TEXT PRIMARY KEY,
                order_id TEXT,
                symbol TEXT NOT NULL,
                check_type TEXT NOT NULL,

                -- Check result
                passed BOOLEAN NOT NULL,
                violation TEXT,

                -- Order details
                order_value REAL,
                order_quantity REAL,

                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_breaches (
                breach_id TEXT PRIMARY KEY,
                breach_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,

                -- Context
                symbol TEXT,
                order_id TEXT,

                -- Resolution
                resolved BOOLEAN DEFAULT 0,
                resolved_at TEXT,
                resolution_notes TEXT,

                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kill_switch_events (
                event_id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                triggered_by TEXT,

                created_at TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def pre_trade_check(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None,
        current_position: dict | None = None,
    ) -> dict:
        """
        Run comprehensive pre-trade risk checks.

        Args:
            order_id: Order ID
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            price: Order price (None for market orders)
            current_position: Current position dict

        Returns:
            Dict with check results and any violations
        """
        violations = []

        # Check 1: Kill switch
        if not self.trading_enabled:
            return {
                "passed": False,
                "violations": [f"Trading halted: {self.kill_switch_reason}"],
                "severity": "CRITICAL",
            }

        # Check 2: Order quantity limit
        if quantity > self.max_order_quantity:
            violations.append(f"Order quantity {quantity} exceeds limit {self.max_order_quantity}")

        # Check 3: Order value limit (if price available)
        if price is not None:
            order_value = quantity * price
            if order_value > self.max_order_value:
                violations.append(
                    f"Order value ${order_value:,.0f} exceeds limit ${self.max_order_value:,.0f}"
                )
        else:
            order_value = None

        # Check 4: Position size limit
        if current_position:
            current_quantity = current_position.get("quantity", 0.0)

            if side == "BUY":
                new_quantity = current_quantity + quantity
            else:  # SELL
                new_quantity = current_quantity - quantity

            if price is not None:
                new_position_value = abs(new_quantity * price)
                if new_position_value > self.max_position_size:
                    violations.append(
                        f"Resulting position value ${new_position_value:,.0f} exceeds limit ${self.max_position_size:,.0f}"
                    )

        # Check 5: Daily trade count
        daily_trades = self._get_daily_trade_count()
        if daily_trades >= self.max_daily_trades:
            violations.append(f"Daily trade limit {self.max_daily_trades} reached")

        # Check 6: Symbol-specific checks
        symbol_check = self._check_symbol_restrictions(symbol)
        if symbol_check:
            violations.append(symbol_check)

        passed = len(violations) == 0

        # Record check
        check = {
            "check_id": f"check_{order_id}",
            "order_id": order_id,
            "symbol": symbol,
            "check_type": "PRE_TRADE",
            "passed": passed,
            "violation": "; ".join(violations) if violations else None,
            "order_value": order_value,
            "order_quantity": quantity,
        }

        self._record_risk_check(check)

        # Log breach if failed
        if not passed:
            self._record_risk_breach(
                breach_type="PRE_TRADE_VIOLATION",
                severity="HIGH",
                description="; ".join(violations),
                symbol=symbol,
                order_id=order_id,
            )

        return {
            "passed": passed,
            "violations": violations,
            "severity": "HIGH" if not passed else None,
            "order_value": order_value,
            "checks_performed": [
                "kill_switch",
                "order_quantity",
                "order_value",
                "position_size",
                "daily_limit",
                "symbol_restrictions",
            ],
        }

    def activate_kill_switch(self, reason: str, triggered_by: str = "system") -> dict:
        """
        Activate kill switch to halt all trading.

        Args:
            reason: Reason for activation
            triggered_by: Who/what triggered it

        Returns:
            Dict with confirmation
        """
        if not self.trading_enabled:
            return {"status": "already_active", "reason": self.kill_switch_reason}

        self.trading_enabled = False
        self.kill_switch_reason = reason

        # Record event
        event = {
            "event_id": f"kill_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "action": "ACTIVATE",
            "reason": reason,
            "triggered_by": triggered_by,
        }

        self._record_kill_switch_event(event)

        # Record breach
        self._record_risk_breach(
            breach_type="KILL_SWITCH_ACTIVATED",
            severity="CRITICAL",
            description=f"Trading halted: {reason}",
            symbol=None,
            order_id=None,
        )

        LOGGER.warning(f"KILL SWITCH ACTIVATED: {reason}")

        return {
            "status": "activated",
            "reason": reason,
            "triggered_by": triggered_by,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def deactivate_kill_switch(self, authorized_by: str = "admin") -> dict:
        """
        Deactivate kill switch to resume trading.

        Args:
            authorized_by: Who authorized deactivation

        Returns:
            Dict with confirmation
        """
        if self.trading_enabled:
            return {"status": "already_inactive", "message": "Trading already enabled"}

        previous_reason = self.kill_switch_reason

        self.trading_enabled = True
        self.kill_switch_reason = None

        # Record event
        event = {
            "event_id": f"kill_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "action": "DEACTIVATE",
            "reason": f"Authorized by {authorized_by}",
            "triggered_by": authorized_by,
        }

        self._record_kill_switch_event(event)

        LOGGER.info(f"KILL SWITCH DEACTIVATED by {authorized_by}")

        return {
            "status": "deactivated",
            "previous_reason": previous_reason,
            "authorized_by": authorized_by,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_kill_switch_status(self) -> dict:
        """Get current kill switch status."""
        return {
            "trading_enabled": self.trading_enabled,
            "kill_switch_active": not self.trading_enabled,
            "reason": self.kill_switch_reason,
            "checked_at": datetime.utcnow().isoformat(),
        }

    def get_risk_limits(self) -> dict:
        """Get current risk limits."""
        return {
            "max_order_value": self.max_order_value,
            "max_position_size": self.max_position_size,
            "max_daily_trades": self.max_daily_trades,
            "max_order_quantity": self.max_order_quantity,
        }

    def update_risk_limits(self, **kwargs) -> dict:
        """
        Update risk limits dynamically.

        Args:
            max_order_value: New max order value
            max_position_size: New max position size
            max_daily_trades: New max daily trades count
            max_order_quantity: New max order quantity

        Returns:
            Dict with updated limits
        """
        if "max_order_value" in kwargs:
            self.max_order_value = kwargs["max_order_value"]
            LOGGER.info(f"Updated risk limit: max_order_value = {kwargs['max_order_value']}")

        if "max_position_size" in kwargs:
            self.max_position_size = kwargs["max_position_size"]
            LOGGER.info(f"Updated risk limit: max_position_size = {kwargs['max_position_size']}")

        if "max_daily_trades" in kwargs:
            self.max_daily_trades = kwargs["max_daily_trades"]
            LOGGER.info(f"Updated risk limit: max_daily_trades = {kwargs['max_daily_trades']}")

        if "max_order_quantity" in kwargs:
            self.max_order_quantity = kwargs["max_order_quantity"]
            LOGGER.info(f"Updated risk limit: max_order_quantity = {kwargs['max_order_quantity']}")

        return self.get_risk_limits()

    def get_recent_breaches(self, limit: int = 10) -> list[dict]:
        """Get recent risk breaches."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM risk_breaches
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (limit,),
            )

            columns = [desc[0] for desc in cursor.description]
            breaches = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]

            conn.close()
            return breaches
        except Exception as e:
            LOGGER.error(f"Failed to get breaches: {e}")
            return []

    def _get_daily_trade_count(self) -> int:
        """Get number of trades today."""
        try:
            today = datetime.utcnow().date().isoformat()

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT COUNT(*) FROM risk_checks
                WHERE DATE(created_at) = ? AND passed = 1
            """,
                (today,),
            )

            count = cursor.fetchone()[0]
            conn.close()

            return count
        except Exception as e:
            LOGGER.error(f"Failed to get daily trade count: {e}")
            return 0

    def _check_symbol_restrictions(self, symbol: str) -> str | None:
        """Check if symbol has trading restrictions."""
        # In production, this would check:
        # - Halted symbols
        # - Restricted securities
        # - Penny stocks
        # - Hard-to-borrow stocks

        # For now, just a placeholder
        restricted_symbols = ["RESTRICTED"]

        if symbol in restricted_symbols:
            return f"Symbol {symbol} is restricted from trading"

        return None

    def _record_risk_check(self, check: dict):
        """Record risk check to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO risk_checks (
                    check_id, order_id, symbol, check_type, passed,
                    violation, order_value, order_quantity, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    check["check_id"],
                    check["order_id"],
                    check["symbol"],
                    check["check_type"],
                    check["passed"],
                    check["violation"],
                    check["order_value"],
                    check["order_quantity"],
                    datetime.utcnow().isoformat(),
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record risk check: {e}")

    def _record_risk_breach(
        self,
        breach_type: str,
        severity: str,
        description: str,
        symbol: str | None = None,
        order_id: str | None = None,
    ):
        """Record risk breach."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            breach_id = f"breach_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            cursor.execute(
                """
                INSERT INTO risk_breaches (
                    breach_id, breach_type, severity, description,
                    symbol, order_id, resolved, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
                (
                    breach_id,
                    breach_type,
                    severity,
                    description,
                    symbol,
                    order_id,
                    datetime.utcnow().isoformat(),
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record risk breach: {e}")

    def _record_kill_switch_event(self, event: dict):
        """Record kill switch event."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO kill_switch_events (
                    event_id, action, reason, triggered_by, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    event["event_id"],
                    event["action"],
                    event["reason"],
                    event["triggered_by"],
                    datetime.utcnow().isoformat(),
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record kill switch event: {e}")


# Singleton instance
_execution_risk: ExecutionRisk | None = None


def get_execution_risk() -> ExecutionRisk:
    """Get singleton execution risk instance."""
    global _execution_risk
    if _execution_risk is None:
        _execution_risk = ExecutionRisk()
    return _execution_risk
