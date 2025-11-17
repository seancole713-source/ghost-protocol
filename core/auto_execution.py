"""
Automated broker execution.
Connects to real brokers (Alpaca, TD Ameritrade) for automated trading.
"""

import os
import time
from typing import Any
import requests


class BrokerExecutor:
    """Base class for broker execution."""
    
    def __init__(self):
        self.enabled = False
        self.api_key = None
        self.api_secret = None
        self.base_url = None
    
    def connect(self) -> dict[str, Any]:
        """Connect to broker API."""
        raise NotImplementedError
    
    def get_account(self) -> dict[str, Any]:
        """Get account information."""
        raise NotImplementedError
    
    def place_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: float,
        order_type: str = "market"
    ) -> dict[str, Any]:
        """Place an order."""
        raise NotImplementedError
    
    def get_positions(self) -> list[dict[str, Any]]:
        """Get current positions."""
        raise NotImplementedError


class AlpacaExecutor(BrokerExecutor):
    """Alpaca broker integration."""
    
    def __init__(self):
        super().__init__()
        # Use existing Railway env var names
        self.api_key = os.getenv("ALPACA_KEY_ID") or os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
        self.base_url = os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.enabled = bool(self.api_key and self.api_secret)
    
    def connect(self) -> dict[str, Any]:
        """Test Alpaca connection."""
        if not self.enabled:
            return {
                "ok": False,
                "error": "Alpaca API credentials not configured"
            }
        
        try:
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            }
            
            response = requests.get(f"{self.base_url}/v2/account", headers=headers, timeout=10)
            
            if response.status_code == 200:
                return {
                    "ok": True,
                    "message": "Connected to Alpaca",
                    "account": response.json()
                }
            else:
                return {
                    "ok": False,
                    "error": f"Connection failed: {response.status_code}"
                }
        
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }
    
    def get_account(self) -> dict[str, Any]:
        """Get Alpaca account info."""
        if not self.enabled:
            return {"ok": False, "error": "Not configured"}
        
        try:
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            }
            
            response = requests.get(f"{self.base_url}/v2/account", headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "ok": True,
                    "cash": float(data.get("cash", 0)),
                    "portfolio_value": float(data.get("portfolio_value", 0)),
                    "buying_power": float(data.get("buying_power", 0)),
                    "equity": float(data.get("equity", 0))
                }
            
            return {"ok": False, "error": f"API error: {response.status_code}"}
        
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market"
    ) -> dict[str, Any]:
        """Place order on Alpaca."""
        if not self.enabled:
            return {"ok": False, "error": "Not configured"}
        
        try:
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            }
            
            payload = {
                "symbol": symbol,
                "qty": quantity,
                "side": side,
                "type": order_type,
                "time_in_force": "day"
            }
            
            response = requests.post(
                f"{self.base_url}/v2/orders",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                order = response.json()
                return {
                    "ok": True,
                    "order_id": order.get("id"),
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "status": order.get("status"),
                    "timestamp": time.time()
                }
            
            return {
                "ok": False,
                "error": f"Order failed: {response.status_code}",
                "details": response.text
            }
        
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def get_positions(self) -> list[dict[str, Any]]:
        """Get Alpaca positions."""
        if not self.enabled:
            return []
        
        try:
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            }
            
            response = requests.get(f"{self.base_url}/v2/positions", headers=headers, timeout=10)
            
            if response.status_code == 200:
                positions = []
                for pos in response.json():
                    positions.append({
                        "symbol": pos.get("symbol"),
                        "quantity": float(pos.get("qty", 0)),
                        "avg_entry_price": float(pos.get("avg_entry_price", 0)),
                        "current_price": float(pos.get("current_price", 0)),
                        "market_value": float(pos.get("market_value", 0)),
                        "unrealized_pnl": float(pos.get("unrealized_pl", 0)),
                        "unrealized_pnl_pct": float(pos.get("unrealized_plpc", 0))
                    })
                return positions
            
            return []
        
        except Exception:
            return []


# Global executor instance
_EXECUTOR: BrokerExecutor | None = None


def get_executor() -> BrokerExecutor:
    """Get broker executor singleton."""
    global _EXECUTOR
    
    if _EXECUTOR is None:
        # Try Alpaca first (easiest to integrate)
        _EXECUTOR = AlpacaExecutor()
    
    return _EXECUTOR


def execute_ghost_signal(
    symbol: str,
    signal: dict[str, Any],
    max_position_pct: float = 0.10
) -> dict[str, Any]:
    """
    Automatically execute a Ghost AI signal.
    
    Args:
        symbol: Stock/crypto ticker
        signal: Ghost signal with action, confidence, position_size_pct
        max_position_pct: Max % of portfolio per position (safety limit)
    
    Returns:
        Execution result
    """
    executor = get_executor()
    
    if not executor.enabled:
        return {
            "ok": False,
            "error": "Broker not configured",
            "message": "Set ALPACA_API_KEY and ALPACA_API_SECRET to enable auto-execution"
        }
    
    action = signal.get("trade_signal") or signal.get("action")
    confidence = signal.get("confidence", 0.0)
    position_size_pct = signal.get("position_size_pct", 0.05)
    
    # Safety checks
    if confidence < 0.85:
        return {
            "ok": False,
            "error": "Confidence too low for auto-execution",
            "confidence": confidence,
            "minimum": 0.85
        }
    
    if position_size_pct > max_position_pct:
        position_size_pct = max_position_pct
    
    # Get account info
    account = executor.get_account()
    if not account.get("ok"):
        return account
    
    buying_power = account.get("buying_power", 0)
    
    # Calculate quantity
    current_price = signal.get("price_now") or signal.get("price_current", 0)
    if not current_price:
        return {"ok": False, "error": "No current price in signal"}
    
    position_value = buying_power * position_size_pct
    quantity = position_value / current_price
    
    # Execute order
    if action == "BUY":
        result = executor.place_order(symbol, "buy", quantity)
    elif action == "SELL":
        result = executor.place_order(symbol, "sell", quantity)
    else:
        return {
            "ok": False,
            "error": f"Unknown action: {action}"
        }
    
    return result
