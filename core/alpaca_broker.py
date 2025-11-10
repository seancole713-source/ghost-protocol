"""
Alpaca Broker Integration for GHOST Trading System
Supports paper trading and live trading with full order management.
"""

import logging
import os
from enum import Enum
from typing import Any

try:
    import requests
except ImportError:
    requests = None

from core.concurrency import AsyncRateLimiter, ConcurrencyMetrics, ExecutionTimer

LOGGER = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderTIF(str, Enum):
    DAY = "day"
    GTC = "gtc"  # Good 'til canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


class OrderStatus(str, Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    PENDING_NEW = "pending_new"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STOPPED = "stopped"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"


class AlpacaBroker:
    """
    Alpaca API integration for stock trading.
    Supports paper trading (default) and live trading.
    """

    def __init__(self):
        self._metrics = ConcurrencyMetrics()
        self.enabled = os.getenv("BROKER", "").lower() == "alpaca"
        self.key_id = os.getenv("ALPACA_KEY_ID", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.paper = os.getenv("ALPACA_PAPER", "1") == "1"
        rate = int(os.getenv("ALPACA_ORDER_RATE", "30"))
        window_s = float(os.getenv("ALPACA_ORDER_WINDOW_S", "60"))
        self._order_limiter = AsyncRateLimiter(rate=rate, per=window_s)

        # Base URL with safety checks
        if self.paper:
            self.base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")
        else:
            self.base_url = os.getenv("APCA_API_BASE_URL", "https://api.alpaca.markets/v2")
            # Extra warning for live trading
            LOGGER.warning("⚠️  LIVE TRADING MODE ENABLED - Real money at risk!")

        # Validate configuration
        if self.enabled and (not self.key_id or not self.secret_key):
            LOGGER.error("Alpaca broker enabled but API keys not configured")
            self.enabled = False

        # Validate URL matches mode
        if self.enabled and self.paper and "paper" not in self.base_url.lower():
            LOGGER.error("SAFETY: Paper mode enabled but URL is not paper-api. Disabling broker.")
            self.enabled = False

        if self.enabled and not self.paper and "paper" in self.base_url.lower():
            LOGGER.error("SAFETY: Live mode enabled but URL is paper-api. Disabling broker.")
            self.enabled = False

        if self.enabled:
            mode = "Paper Trading" if self.paper else "LIVE TRADING"
            LOGGER.info(f"Alpaca broker initialized - Mode: {mode}, URL: {self.base_url}")

    def _headers(self) -> dict[str, str]:
        """Get headers for Alpaca API requests."""
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        """Make authenticated request to Alpaca API with enhanced error handling."""
        if not self.enabled:
            raise Exception("Alpaca broker not enabled. Set BROKER=alpaca and configure API keys.")

        if not requests:
            raise Exception("requests library not available. Install with: pip install requests")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._headers()

        try:
            # Apply rate limiting for write operations
            if method.upper() in {"POST", "PATCH", "DELETE"}:
                self._order_limiter.blocking_acquire()

            # Execute request with timing metrics
            with ExecutionTimer(
                f"alpaca:{method}:{endpoint}", logger=LOGGER, metrics=self._metrics
            ):
                response = requests.request(method, url, headers=headers, timeout=10, **kwargs)

            # Raise for HTTP errors
            response.raise_for_status()

            # Parse response
            return response.json() if response.text else {}

        except requests.exceptions.Timeout as e:
            error_msg = f"Alpaca API timeout after 10s: {endpoint}"
            LOGGER.error(error_msg)
            raise Exception(error_msg) from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "unknown"
            error_detail = ""

            try:
                error_json = e.response.json() if e.response else {}
                error_detail = error_json.get("message", e.response.text)
            except Exception:
                error_detail = e.response.text if e.response else str(e)

            # Provide helpful error messages
            if status_code == 401:
                error_msg = "Alpaca authentication failed. Check API keys."
            elif status_code == 403:
                error_msg = f"Alpaca access forbidden: {error_detail}"
            elif status_code == 404:
                error_msg = f"Alpaca resource not found: {endpoint}"
            elif status_code == 422:
                error_msg = f"Alpaca validation error: {error_detail}"
            elif status_code == 429:
                error_msg = "Alpaca rate limit exceeded. Slow down requests."
            else:
                error_msg = f"Alpaca API error [{status_code}]: {error_detail}"

            LOGGER.error(error_msg)
            raise Exception(error_msg) from e

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Cannot connect to Alpaca API: {url}. Check network/DNS."
            LOGGER.error(error_msg)
            raise Exception(error_msg) from e

        except Exception as e:
            error_msg = f"Alpaca request failed [{method} {endpoint}]: {e}"
            LOGGER.error(error_msg)
            raise Exception(error_msg) from e

    def get_account(self) -> dict[str, Any]:
        """Get account information."""
        return self._request("GET", "/account")

    def get_positions(self) -> list[dict[str, Any]]:
        """Get all open positions."""
        return self._request("GET", "/positions")

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Get position for a specific symbol."""
        try:
            return self._request("GET", f"/positions/{symbol}")
        except Exception:
            return None

    def close_position(self, symbol: str) -> dict[str, Any]:
        """Close entire position for a symbol."""
        return self._request("DELETE", f"/positions/{symbol}")

    def submit_order(
        self,
        symbol: str,
        qty: float | None = None,
        notional: float | None = None,
        side: OrderSide = OrderSide.BUY,
        type: OrderType = OrderType.MARKET,
        time_in_force: OrderTIF = OrderTIF.DAY,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        extended_hours: bool = False,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Submit a new order.

        Args:
            symbol: Stock symbol (e.g., "WOLF", "AAPL")
            qty: Number of shares (use qty OR notional, not both)
            notional: Dollar amount to trade (fractional shares)
            side: "buy" or "sell"
            type: "market", "limit", "stop", "stop_limit", "trailing_stop"
            time_in_force: "day", "gtc", "ioc", "fok"
            limit_price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            trail_price: Trail amount in dollars (for trailing stop)
            trail_percent: Trail percent (for trailing stop)
            extended_hours: Allow extended hours trading
            client_order_id: Custom order ID for tracking

        Returns:
            Order object with order ID, status, etc.
        """
        # PRE-FLIGHT CHECKS
        if not qty and not notional:
            raise ValueError("Must specify either qty or notional")
        if qty and notional:
            raise ValueError("Cannot specify both qty and notional")

        # Validate order type requirements
        if type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and limit_price is None:
            raise ValueError(f"limit_price required for {type} orders")

        if type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            raise ValueError(f"stop_price required for {type} orders")

        if type == OrderType.TRAILING_STOP and trail_price is None and trail_percent is None:
            raise ValueError(
                "Either trail_price or trail_percent required for trailing_stop orders"
            )

        # Log order details (especially for live trading)
        mode = "PAPER" if self.paper else "LIVE"
        side_str = side.value if isinstance(side, OrderSide) else side
        type_str = type.value if isinstance(type, OrderType) else type
        qty_str = f"{qty} shares" if qty else f"${notional} notional"

        LOGGER.info(
            f"[{mode}] Submitting order: {side_str.upper()} {qty_str} {symbol} ({type_str})"
        )

        if not self.paper:
            LOGGER.warning(
                f"⚠️  LIVE ORDER: {side_str.upper()} {qty_str} {symbol} - Real money at risk!"
            )

        payload = {
            "symbol": symbol.upper(),
            "side": side.value if isinstance(side, OrderSide) else side,
            "type": type.value if isinstance(type, OrderType) else type,
            "time_in_force": time_in_force.value
            if isinstance(time_in_force, OrderTIF)
            else time_in_force,
        }

        if qty is not None:
            payload["qty"] = qty
        if notional is not None:
            payload["notional"] = notional
        if limit_price is not None:
            payload["limit_price"] = limit_price
        if stop_price is not None:
            payload["stop_price"] = stop_price
        if trail_price is not None:
            payload["trail_price"] = trail_price
        if trail_percent is not None:
            payload["trail_percent"] = trail_percent
        if extended_hours:
            payload["extended_hours"] = True
        if client_order_id:
            payload["client_order_id"] = client_order_id

        result = self._request("POST", "/orders", json=payload)

        # Log successful submission with order ID
        order_id = result.get("id", "unknown")
        status = result.get("status", "unknown")
        LOGGER.info(f"[{mode}] Order submitted successfully: ID={order_id}, status={status}")

        return result

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order by ID."""
        return self._request("GET", f"/orders/{order_id}")

    def get_orders(
        self,
        status: str | None = None,
        limit: int = 50,
        after: str | None = None,
        until: str | None = None,
        direction: str = "desc",
        nested: bool = True,
        symbols: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get orders with filters.

        Args:
            status: "open", "closed", "all"
            limit: Max number of orders to return (default 50, max 500)
            after: Filter by orders after this timestamp
            until: Filter by orders until this timestamp
            direction: "asc" or "desc"
            nested: Show nested orders
            symbols: Comma-separated list of symbols
        """
        params = {
            "limit": limit,
            "direction": direction,
            "nested": nested,
        }
        if status:
            params["status"] = status
        if after:
            params["after"] = after
        if until:
            params["until"] = until
        if symbols:
            params["symbols"] = symbols

        return self._request("GET", "/orders", params=params)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order by ID."""
        return self._request("DELETE", f"/orders/{order_id}")

    def cancel_all_orders(self) -> list[dict[str, Any]]:
        """Cancel all open orders."""
        return self._request("DELETE", "/orders")

    def replace_order(
        self,
        order_id: str,
        qty: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail: float | None = None,
        time_in_force: OrderTIF | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Replace an existing order."""
        payload = {}
        if qty is not None:
            payload["qty"] = qty
        if limit_price is not None:
            payload["limit_price"] = limit_price
        if stop_price is not None:
            payload["stop_price"] = stop_price
        if trail is not None:
            payload["trail"] = trail
        if time_in_force:
            payload["time_in_force"] = (
                time_in_force.value if isinstance(time_in_force, OrderTIF) else time_in_force
            )
        if client_order_id:
            payload["client_order_id"] = client_order_id

        return self._request("PATCH", f"/orders/{order_id}", json=payload)

    def get_clock(self) -> dict[str, Any]:
        """Get market clock info (is market open, next open/close times)."""
        return self._request("GET", "/clock")

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            clock = self.get_clock()
            return clock.get("is_open", False)
        except Exception as e:
            LOGGER.error(f"Failed to check market status: {e}")
            return False

    def get_calendar(
        self, start: str | None = None, end: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Get market calendar (trading days).

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        """
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        return self._request("GET", "/calendar", params=params)

    def health_check(self) -> dict[str, Any]:
        """
        Check broker health and connectivity.

        Returns:
            dict with "ok", "message", and account info
        """
        try:
            account = self.get_account()
            positions = self.get_positions()
            clock = self.get_clock()

            return {
                "ok": True,
                "broker": "alpaca",
                "paper": self.paper,
                "account_id": account.get("id"),
                "account_number": account.get("account_number"),
                "status": account.get("status"),
                "buying_power": float(account.get("buying_power", 0)),
                "cash": float(account.get("cash", 0)),
                "portfolio_value": float(account.get("portfolio_value", 0)),
                "positions_count": len(positions),
                "market_open": clock.get("is_open", False),
                "timestamp": clock.get("timestamp"),
            }
        except Exception as e:
            return {
                "ok": False,
                "broker": "alpaca",
                "error": str(e),
            }

    def metrics_snapshot(self) -> dict[str, Any]:
        data = self._metrics.snapshot()
        data["enabled"] = self.enabled
        data["paper"] = self.paper
        return data


# Global instance
_broker_instance = None


def get_broker() -> AlpacaBroker:
    """Get global Alpaca broker instance."""
    global _broker_instance
    if _broker_instance is None:
        _broker_instance = AlpacaBroker()
    return _broker_instance
