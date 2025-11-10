"""
Stage 5: Order Manager
Advanced Execution & Order Management

Features: order lifecycle management, multiple order types, position tracking, execution simulation.
"""

import logging
import sqlite3
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the system."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order side (buy or sell)."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order lifecycle status."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """Order time in force."""

    DAY = "DAY"  # Good for day
    GTC = "GTC"  # Good till cancelled
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill


class OrderManager:
    """
    Advanced order management system.

    Features:
    - Multiple order types (market, limit, stop, stop-limit)
    - Order lifecycle tracking
    - Position management
    - Execution simulation
    - Fill tracking with partial fills
    """

    def __init__(self, db_path: str = "data/order_manager.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.active_orders: dict[str, dict] = {}
        self.positions: dict[str, dict] = {}

        self._init_db()
        self._load_active_orders()
        self._load_positions()

        LOGGER.info(f"Order manager initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database for order management."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                client_order_id TEXT,
                symbol TEXT NOT NULL,
                order_type TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL,
                stop_price REAL,
                time_in_force TEXT NOT NULL,

                -- Status
                status TEXT NOT NULL,
                filled_quantity REAL DEFAULT 0.0,
                avg_fill_price REAL,

                -- Timestamps
                created_at TEXT NOT NULL,
                submitted_at TEXT,
                filled_at TEXT,
                cancelled_at TEXT,

                -- Metadata
                strategy TEXT,
                notes TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                commission REAL DEFAULT 0.0,
                filled_at TEXT NOT NULL,

                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity REAL NOT NULL,
                avg_cost REAL NOT NULL,
                market_value REAL,
                unrealized_pnl REAL,
                realized_pnl REAL DEFAULT 0.0,
                updated_at TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: float,
        price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        strategy: str | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        """
        Create a new order.

        Args:
            symbol: Trading symbol
            order_type: OrderType enum
            side: OrderSide enum (BUY/SELL)
            quantity: Number of shares
            price: Limit price (for LIMIT/STOP_LIMIT)
            stop_price: Stop price (for STOP/STOP_LIMIT)
            time_in_force: TimeInForce enum
            strategy: Strategy name
            client_order_id: Optional client ID

        Returns:
            Dict with order details
        """
        # Validation
        if quantity <= 0:
            return {"error": "Quantity must be positive"}

        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            return {"error": f"{order_type.value} orders require price"}

        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            return {"error": f"{order_type.value} orders require stop_price"}

        # Generate order ID
        order_id = str(uuid.uuid4())[:8]

        order = {
            "order_id": order_id,
            "client_order_id": client_order_id or order_id,
            "symbol": symbol,
            "order_type": order_type.value,
            "side": side.value,
            "quantity": quantity,
            "price": price,
            "stop_price": stop_price,
            "time_in_force": time_in_force.value,
            "status": OrderStatus.PENDING.value,
            "filled_quantity": 0.0,
            "avg_fill_price": None,
            "created_at": datetime.utcnow().isoformat(),
            "submitted_at": None,
            "filled_at": None,
            "cancelled_at": None,
            "strategy": strategy,
            "notes": None,
        }

        # Store in memory and database
        self.active_orders[order_id] = order
        self._record_order(order)

        LOGGER.info(
            f"Order created: {order_id} {side.value} {quantity} {symbol} @ {price or 'MARKET'}"
        )

        return {"status": "created", "order_id": order_id, "order": order}

    def submit_order(self, order_id: str) -> dict:
        """
        Submit order for execution (simulated).

        In production, this would send to broker API.
        For Stage 5, we simulate immediate execution.
        """
        if order_id not in self.active_orders:
            return {"error": f"Order {order_id} not found"}

        order = self.active_orders[order_id]

        if order["status"] != OrderStatus.PENDING.value:
            return {"error": f"Order {order_id} already {order['status']}"}

        # Update status
        order["status"] = OrderStatus.SUBMITTED.value
        order["submitted_at"] = datetime.utcnow().isoformat()
        self._update_order_status(order_id, OrderStatus.SUBMITTED, order["submitted_at"])

        # Simulate execution for MARKET orders (instant fill)
        if order["order_type"] == OrderType.MARKET.value:
            # Simulate fill at current market price (mock)
            simulated_price = order["price"] or 100.0  # In production, fetch real price
            self._fill_order(order_id, order["quantity"], simulated_price)

        return {"status": "submitted", "order_id": order_id, "order": order}

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an active order."""
        if order_id not in self.active_orders:
            return {"error": f"Order {order_id} not found"}

        order = self.active_orders[order_id]

        if order["status"] in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value]:
            return {"error": f"Order {order_id} already {order['status']}"}

        order["status"] = OrderStatus.CANCELLED.value
        order["cancelled_at"] = datetime.utcnow().isoformat()

        self._update_order_status(order_id, OrderStatus.CANCELLED, order["cancelled_at"])

        # Remove from active orders
        del self.active_orders[order_id]

        LOGGER.info(f"Order cancelled: {order_id}")

        return {"status": "cancelled", "order_id": order_id, "order": order}

    def _fill_order(self, order_id: str, quantity: float, price: float, commission: float = 0.0):
        """
        Fill an order (full or partial).

        Args:
            order_id: Order ID
            quantity: Quantity filled
            price: Fill price
            commission: Commission charged
        """
        if order_id not in self.active_orders:
            return

        order = self.active_orders[order_id]

        # Record fill
        fill_id = str(uuid.uuid4())[:8]
        fill = {
            "fill_id": fill_id,
            "order_id": order_id,
            "symbol": order["symbol"],
            "side": order["side"],
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "filled_at": datetime.utcnow().isoformat(),
        }

        self._record_fill(fill)

        # Update order
        previous_filled = order["filled_quantity"]
        order["filled_quantity"] += quantity

        # Calculate average fill price
        if order["avg_fill_price"] is None:
            order["avg_fill_price"] = price
        else:
            total_filled = order["filled_quantity"]
            order["avg_fill_price"] = (
                order["avg_fill_price"] * previous_filled + price * quantity
            ) / total_filled

        # Update status
        if order["filled_quantity"] >= order["quantity"]:
            order["status"] = OrderStatus.FILLED.value
            order["filled_at"] = datetime.utcnow().isoformat()
            self._update_order_status(
                order_id, OrderStatus.FILLED, order["filled_at"], order["avg_fill_price"]
            )

            # Remove from active orders
            del self.active_orders[order_id]
        else:
            order["status"] = OrderStatus.PARTIAL.value
            self._update_order_status(order_id, OrderStatus.PARTIAL, None, order["avg_fill_price"])

        # Update position
        self._update_position(order["symbol"], order["side"], quantity, price)

        LOGGER.info(
            f"Order filled: {order_id} {quantity} @ {price} (total: {order['filled_quantity']}/{order['quantity']})"
        )

    def _update_position(self, symbol: str, side: str, quantity: float, price: float):
        """Update position after fill."""
        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "quantity": 0.0,
                "avg_cost": 0.0,
                "market_value": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "updated_at": datetime.utcnow().isoformat(),
            }

        position = self.positions[symbol]

        if side == OrderSide.BUY.value:
            # Add to position
            total_cost = position["avg_cost"] * position["quantity"] + price * quantity
            position["quantity"] += quantity
            position["avg_cost"] = (
                total_cost / position["quantity"] if position["quantity"] > 0 else 0.0
            )
        else:  # SELL
            # Reduce position and calculate realized P&L
            if position["quantity"] > 0:
                realized = (price - position["avg_cost"]) * min(quantity, position["quantity"])
                position["realized_pnl"] += realized

            position["quantity"] -= quantity

            # If position flipped to short, reset avg cost
            if position["quantity"] < 0 and position["quantity"] + quantity >= 0:
                position["avg_cost"] = price

        position["updated_at"] = datetime.utcnow().isoformat()

        self._record_position(position)

    def get_order(self, order_id: str) -> dict | None:
        """Get order details."""
        if order_id in self.active_orders:
            return self.active_orders[order_id]

        # Check database for historical order
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row, strict=False))

            return None
        except Exception as e:
            LOGGER.error(f"Failed to get order: {e}")
            return None

    def get_active_orders(self, symbol: str | None = None) -> list[dict]:
        """Get all active orders, optionally filtered by symbol."""
        orders = list(self.active_orders.values())

        if symbol:
            orders = [o for o in orders if o["symbol"] == symbol]

        return orders

    def get_position(self, symbol: str) -> dict | None:
        """Get position for symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> list[dict]:
        """Get all positions."""
        return list(self.positions.values())

    def get_order_history(self, symbol: str | None = None, limit: int = 100) -> list[dict]:
        """Get order history."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            if symbol:
                cursor.execute(
                    "SELECT * FROM orders WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                cursor.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,))

            columns = [desc[0] for desc in cursor.description]
            orders = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]

            conn.close()
            return orders
        except Exception as e:
            LOGGER.error(f"Failed to get order history: {e}")
            return []

    def _record_order(self, order: dict):
        """Record order to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO orders (
                    order_id, client_order_id, symbol, order_type, side, quantity,
                    price, stop_price, time_in_force, status, filled_quantity,
                    avg_fill_price, created_at, submitted_at, filled_at,
                    cancelled_at, strategy, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    order["order_id"],
                    order["client_order_id"],
                    order["symbol"],
                    order["order_type"],
                    order["side"],
                    order["quantity"],
                    order["price"],
                    order["stop_price"],
                    order["time_in_force"],
                    order["status"],
                    order["filled_quantity"],
                    order["avg_fill_price"],
                    order["created_at"],
                    order["submitted_at"],
                    order["filled_at"],
                    order["cancelled_at"],
                    order["strategy"],
                    order["notes"],
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record order: {e}")

    def _update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        timestamp: str | None = None,
        avg_fill_price: float | None = None,
    ):
        """Update order status in database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            if status == OrderStatus.SUBMITTED:
                cursor.execute(
                    "UPDATE orders SET status = ?, submitted_at = ? WHERE order_id = ?",
                    (status.value, timestamp, order_id),
                )
            elif status == OrderStatus.FILLED:
                cursor.execute(
                    "UPDATE orders SET status = ?, filled_at = ?, avg_fill_price = ? WHERE order_id = ?",
                    (status.value, timestamp, avg_fill_price, order_id),
                )
            elif status == OrderStatus.CANCELLED:
                cursor.execute(
                    "UPDATE orders SET status = ?, cancelled_at = ? WHERE order_id = ?",
                    (status.value, timestamp, order_id),
                )
            elif status == OrderStatus.PARTIAL:
                cursor.execute(
                    "UPDATE orders SET status = ?, avg_fill_price = ? WHERE order_id = ?",
                    (status.value, avg_fill_price, order_id),
                )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to update order status: {e}")

    def _record_fill(self, fill: dict):
        """Record fill to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO fills (
                    fill_id, order_id, symbol, side, quantity, price, commission, filled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    fill["fill_id"],
                    fill["order_id"],
                    fill["symbol"],
                    fill["side"],
                    fill["quantity"],
                    fill["price"],
                    fill["commission"],
                    fill["filled_at"],
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record fill: {e}")

    def _record_position(self, position: dict):
        """Record/update position in database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO positions (
                    symbol, quantity, avg_cost, market_value, unrealized_pnl, realized_pnl, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    position["symbol"],
                    position["quantity"],
                    position["avg_cost"],
                    position["market_value"],
                    position["unrealized_pnl"],
                    position["realized_pnl"],
                    position["updated_at"],
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record position: {e}")

    def _load_active_orders(self):
        """Load active orders from database on startup."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM orders WHERE status IN ('PENDING', 'SUBMITTED', 'PARTIAL')
            """)

            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                order = dict(zip(columns, row, strict=False))
                self.active_orders[order["order_id"]] = order

            conn.close()

            LOGGER.info(f"Loaded {len(self.active_orders)} active orders")
        except Exception as e:
            LOGGER.error(f"Failed to load active orders: {e}")

    def _load_positions(self):
        """Load positions from database on startup."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM positions")

            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                position = dict(zip(columns, row, strict=False))
                self.positions[position["symbol"]] = position

            conn.close()

            LOGGER.info(f"Loaded {len(self.positions)} positions")
        except Exception as e:
            LOGGER.error(f"Failed to load positions: {e}")

    def create_trailing_stop(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        trail_percent: float,
        trail_amount: float | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        strategy: str | None = None,
    ) -> dict:
        """
        Create a trailing stop order that adjusts stop price as market moves favorably.

        Args:
            symbol: Trading symbol
            side: Order side (SELL for long positions, BUY for short positions)
            quantity: Order quantity
            trail_percent: Trailing percentage (e.g., 5.0 for 5%)
            trail_amount: Trailing dollar amount (optional, overrides trail_percent)
            time_in_force: Order time in force
            strategy: Strategy name for tracking

        Returns:
            Order dictionary with trailing stop parameters
        """
        order_id = f"ORD_{uuid.uuid4().hex[:12].upper()}"

        order = {
            "order_id": order_id,
            "client_order_id": None,
            "symbol": symbol,
            "order_type": "TRAILING_STOP",
            "side": side.value,
            "quantity": quantity,
            "price": None,
            "stop_price": None,  # Will be calculated on first update
            "trail_percent": trail_percent,
            "trail_amount": trail_amount,
            "highest_price": None,  # Track highest price for long positions
            "lowest_price": None,  # Track lowest price for short positions
            "time_in_force": time_in_force.value,
            "status": OrderStatus.PENDING.value,
            "filled_quantity": 0.0,
            "avg_fill_price": None,
            "created_at": datetime.now().isoformat(),
            "submitted_at": None,
            "filled_at": None,
            "cancelled_at": None,
            "strategy": strategy,
            "notes": f"Trailing stop: {trail_percent}%"
            if trail_amount is None
            else f"Trailing stop: ${trail_amount}",
        }

        self.active_orders[order_id] = order
        self._record_order(order)

        LOGGER.info(f"Created trailing stop order {order_id} for {symbol}: {side.value} {quantity}")
        return order

    def update_trailing_stops(self, current_prices: dict[str, float]):
        """
        Update all active trailing stop orders based on current market prices.

        Args:
            current_prices: Dictionary of {symbol: current_price}

        Returns:
            List of triggered orders
        """
        triggered_orders = []

        for order_id, order in list(self.active_orders.items()):
            if order["order_type"] != "TRAILING_STOP":
                continue

            symbol = order["symbol"]
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            side = order["side"]
            trail_percent = order.get("trail_percent")
            trail_amount = order.get("trail_amount")

            # Validate trail parameters to prevent None division
            if trail_amount is None and trail_percent is None:
                LOGGER.warning(
                    f"Trailing stop {order_id} missing trail_amount and trail_percent, skipping update"
                )
                continue
            if trail_percent is not None and trail_percent <= 0:
                LOGGER.error(
                    f"Trailing stop {order_id} has invalid trail_percent={trail_percent}, must be positive"
                )
                continue
            if trail_amount is not None and trail_amount <= 0:
                LOGGER.error(
                    f"Trailing stop {order_id} has invalid trail_amount={trail_amount}, must be positive"
                )
                continue

            # Initialize tracking prices on first update
            if order["highest_price"] is None:
                order["highest_price"] = current_price
                order["lowest_price"] = current_price

            # Update tracking prices
            if current_price > order["highest_price"]:
                order["highest_price"] = current_price
            if current_price < order["lowest_price"]:
                order["lowest_price"] = current_price

            # Calculate stop price based on trail type
            if side == "SELL":  # Long position - trail below market
                reference_price = order["highest_price"]

                if trail_amount:
                    new_stop_price = reference_price - trail_amount
                else:
                    assert trail_percent is not None  # Validated above
                    new_stop_price = reference_price * (1 - trail_percent / 100)

                # Update stop price if it moved up
                if order["stop_price"] is None or new_stop_price > order["stop_price"]:
                    order["stop_price"] = new_stop_price
                    LOGGER.info(f"Trailing stop {order_id} updated: new stop ${new_stop_price:.2f}")

                # Check if stop triggered
                if current_price <= order["stop_price"]:
                    LOGGER.info(f"Trailing stop {order_id} triggered at ${current_price:.2f}")
                    self._fill_order(order_id, order["quantity"], current_price)
                    triggered_orders.append(order)

            elif side == "BUY":  # Short position - trail above market
                reference_price = order["lowest_price"]

                if trail_amount:
                    new_stop_price = reference_price + trail_amount
                else:
                    assert trail_percent is not None  # Validated above
                    new_stop_price = reference_price * (1 + trail_percent / 100)

                # Update stop price if it moved down
                if order["stop_price"] is None or new_stop_price < order["stop_price"]:
                    order["stop_price"] = new_stop_price
                    LOGGER.info(f"Trailing stop {order_id} updated: new stop ${new_stop_price:.2f}")

                # Check if stop triggered
                if current_price >= order["stop_price"]:
                    LOGGER.info(f"Trailing stop {order_id} triggered at ${current_price:.2f}")
                    self._fill_order(order_id, order["quantity"], current_price)
                    triggered_orders.append(order)

        return triggered_orders

    def get_trailing_stops(self, symbol: str | None = None) -> list[dict]:
        """
        Get all active trailing stop orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of trailing stop orders
        """
        trailing_stops = [
            order
            for order in self.active_orders.values()
            if order["order_type"] == "TRAILING_STOP"
            and (symbol is None or order["symbol"] == symbol)
        ]
        return trailing_stops


# Singleton instance
_order_manager: OrderManager | None = None


def get_order_manager() -> OrderManager:
    """Get singleton order manager instance."""
    global _order_manager
    if _order_manager is None:
        _order_manager = OrderManager()
    return _order_manager
