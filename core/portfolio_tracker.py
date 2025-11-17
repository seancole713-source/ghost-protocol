"""
Portfolio tracking and P&L management.
Tracks user positions, calculates gains/losses, and manages risk exposure.
"""

from typing import Any
import time
import os

# In-memory portfolio (production would use database)
_PORTFOLIO: dict[str, dict[str, Any]] = {}


def add_position(symbol: str, quantity: float, entry_price: float, timestamp: float | None = None) -> dict[str, Any]:
    """
    Add or update a position in the portfolio.
    
    Args:
        symbol: Stock/crypto ticker
        quantity: Number of shares/tokens
        entry_price: Average cost basis
        timestamp: Entry time (defaults to now)
    
    Returns:
        Updated position data
    """
    if timestamp is None:
        timestamp = time.time()
    
    if symbol in _PORTFOLIO:
        # Average down/up existing position
        old_qty = _PORTFOLIO[symbol]["quantity"]
        old_price = _PORTFOLIO[symbol]["entry_price"]
        
        total_qty = old_qty + quantity
        avg_price = ((old_qty * old_price) + (quantity * entry_price)) / total_qty
        
        _PORTFOLIO[symbol] = {
            "symbol": symbol,
            "quantity": total_qty,
            "entry_price": avg_price,
            "first_entry": _PORTFOLIO[symbol]["first_entry"],
            "last_update": timestamp,
            "trades": _PORTFOLIO[symbol].get("trades", 1) + 1
        }
    else:
        _PORTFOLIO[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "first_entry": timestamp,
            "last_update": timestamp,
            "trades": 1
        }
    
    return _PORTFOLIO[symbol]


def remove_position(symbol: str, quantity: float | None = None) -> dict[str, Any]:
    """
    Remove or reduce a position.
    
    Args:
        symbol: Stock/crypto ticker
        quantity: Amount to sell (None = sell all)
    
    Returns:
        Realized P&L info
    """
    if symbol not in _PORTFOLIO:
        return {"ok": False, "error": "Position not found"}
    
    position = _PORTFOLIO[symbol]
    
    if quantity is None or quantity >= position["quantity"]:
        # Sell entire position
        removed = _PORTFOLIO.pop(symbol)
        return {
            "ok": True,
            "action": "closed",
            "symbol": symbol,
            "quantity": removed["quantity"],
            "entry_price": removed["entry_price"]
        }
    else:
        # Partial sell
        _PORTFOLIO[symbol]["quantity"] -= quantity
        _PORTFOLIO[symbol]["last_update"] = time.time()
        return {
            "ok": True,
            "action": "reduced",
            "symbol": symbol,
            "quantity_sold": quantity,
            "quantity_remaining": _PORTFOLIO[symbol]["quantity"]
        }


def get_portfolio() -> dict[str, Any]:
    """Get current portfolio with P&L calculations."""
    return {
        "positions": list(_PORTFOLIO.values()),
        "total_positions": len(_PORTFOLIO),
        "symbols": list(_PORTFOLIO.keys())
    }


def calculate_pnl(symbol: str, current_price: float) -> dict[str, Any]:
    """
    Calculate P&L for a specific position.
    
    Returns:
        {
            "symbol": str,
            "quantity": float,
            "entry_price": float,
            "current_price": float,
            "cost_basis": float,
            "current_value": float,
            "unrealized_pnl": float,
            "unrealized_pnl_pct": float,
            "hold_time_days": float
        }
    """
    if symbol not in _PORTFOLIO:
        return {"ok": False, "error": "No position found"}
    
    position = _PORTFOLIO[symbol]
    quantity = position["quantity"]
    entry_price = position["entry_price"]
    
    cost_basis = quantity * entry_price
    current_value = quantity * current_price
    unrealized_pnl = current_value - cost_basis
    unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
    
    hold_time_days = (time.time() - position["first_entry"]) / 86400
    
    return {
        "ok": True,
        "symbol": symbol,
        "quantity": quantity,
        "entry_price": round(entry_price, 2),
        "current_price": round(current_price, 2),
        "cost_basis": round(cost_basis, 2),
        "current_value": round(current_value, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
        "hold_time_days": round(hold_time_days, 1),
        "trades": position.get("trades", 1)
    }


def get_portfolio_summary(current_prices: dict[str, float]) -> dict[str, Any]:
    """
    Get complete portfolio summary with total P&L.
    
    Args:
        current_prices: {symbol: price} mapping
    
    Returns:
        Complete portfolio metrics
    """
    total_cost = 0
    total_value = 0
    positions_pnl = []
    
    for symbol, position in _PORTFOLIO.items():
        current_price = current_prices.get(symbol, position["entry_price"])
        pnl = calculate_pnl(symbol, current_price)
        
        if pnl.get("ok"):
            total_cost += pnl["cost_basis"]
            total_value += pnl["current_value"]
            positions_pnl.append(pnl)
    
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    
    # Calculate winners/losers
    winners = [p for p in positions_pnl if p["unrealized_pnl"] > 0]
    losers = [p for p in positions_pnl if p["unrealized_pnl"] < 0]
    
    return {
        "ok": True,
        "total_positions": len(_PORTFOLIO),
        "total_cost_basis": round(total_cost, 2),
        "total_current_value": round(total_value, 2),
        "total_unrealized_pnl": round(total_pnl, 2),
        "total_unrealized_pnl_pct": round(total_pnl_pct, 2),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(len(winners) / len(positions_pnl) * 100, 1) if positions_pnl else 0,
        "positions": positions_pnl,
        "top_gainer": max(positions_pnl, key=lambda x: x["unrealized_pnl_pct"]) if positions_pnl else None,
        "top_loser": min(positions_pnl, key=lambda x: x["unrealized_pnl_pct"]) if positions_pnl else None
    }
