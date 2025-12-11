"""
Phase 6: Real-Time Trade Monitoring System
Provides WebSocket streaming, P&L tracking, and performance metrics.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta, UTC
from typing import Any, Optional
from collections import defaultdict, deque

LOGGER = logging.getLogger(__name__)

# Global state for real-time monitoring
_trade_history: deque = deque(maxlen=1000)  # Last 1000 trades
_active_positions: dict[str, dict] = {}
_performance_metrics: dict[str, Any] = {
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "total_pnl": 0.0,
    "daily_pnl": 0.0,
    "peak_portfolio_value": 100000.0,
    "current_drawdown": 0.0,
    "max_drawdown": 0.0,
    "last_reset": datetime.now(UTC).isoformat()
}
_websocket_subscribers: set = set()


def record_trade(trade_data: dict[str, Any]) -> None:
    """
    Record a trade execution for monitoring and analytics.
    
    Args:
        trade_data: Dict with symbol, side, quantity, price, timestamp, etc.
    """
    global _trade_history, _active_positions, _performance_metrics
    
    try:
        symbol = trade_data.get("symbol", "UNKNOWN")
        side = trade_data.get("side", "buy").lower()
        quantity = float(trade_data.get("quantity", 0))
        price = float(trade_data.get("price", 0))
        timestamp = trade_data.get("timestamp", datetime.now(UTC).isoformat())
        
        # Add to trade history
        trade_record = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "value": quantity * price,
            "timestamp": timestamp,
            "status": trade_data.get("status", "executed")
        }
        _trade_history.append(trade_record)
        
        # Update active positions
        if side == "buy":
            if symbol not in _active_positions:
                _active_positions[symbol] = {
                    "quantity": 0,
                    "avg_price": 0.0,
                    "total_cost": 0.0
                }
            pos = _active_positions[symbol]
            total_cost = pos["total_cost"] + (quantity * price)
            total_qty = pos["quantity"] + quantity
            pos["quantity"] = total_qty
            pos["avg_price"] = total_cost / total_qty if total_qty > 0 else 0.0
            pos["total_cost"] = total_cost
            
        elif side == "sell":
            if symbol in _active_positions:
                pos = _active_positions[symbol]
                # Calculate P&L for this sell
                pnl = (price - pos["avg_price"]) * quantity
                _performance_metrics["total_pnl"] += pnl
                _performance_metrics["daily_pnl"] += pnl
                
                # Update win/loss counters
                _performance_metrics["total_trades"] += 1
                if pnl > 0:
                    _performance_metrics["winning_trades"] += 1
                else:
                    _performance_metrics["losing_trades"] += 1
                
                # Update position
                pos["quantity"] -= quantity
                pos["total_cost"] -= pos["avg_price"] * quantity
                
                # Remove if fully closed
                if pos["quantity"] <= 0:
                    del _active_positions[symbol]
        
        # Update drawdown
        _update_drawdown()
        
        # Broadcast to WebSocket subscribers
        asyncio.create_task(_broadcast_trade_update(trade_record))
        
        LOGGER.info(f"[TRADE-MONITOR] Recorded {side.upper()} {quantity} {symbol} @ ${price:.2f}")
        
    except Exception as e:
        LOGGER.error(f"[TRADE-MONITOR] Failed to record trade: {e}", exc_info=True)


def _update_drawdown() -> None:
    """Update current and maximum drawdown metrics."""
    global _performance_metrics
    
    current_value = 100000.0 + _performance_metrics["total_pnl"]
    peak = _performance_metrics["peak_portfolio_value"]
    
    if current_value > peak:
        _performance_metrics["peak_portfolio_value"] = current_value
        _performance_metrics["current_drawdown"] = 0.0
    else:
        drawdown = (peak - current_value) / peak if peak > 0 else 0.0
        _performance_metrics["current_drawdown"] = drawdown
        if drawdown > _performance_metrics["max_drawdown"]:
            _performance_metrics["max_drawdown"] = drawdown


def get_trade_history(limit: int = 100) -> list[dict]:
    """Get recent trade history."""
    return list(_trade_history)[-limit:]


def get_active_positions() -> dict[str, dict]:
    """Get current active positions."""
    return dict(_active_positions)


def get_performance_metrics() -> dict[str, Any]:
    """Get current performance metrics."""
    metrics = dict(_performance_metrics)
    
    # Calculate win rate
    total = metrics["total_trades"]
    metrics["win_rate"] = (metrics["winning_trades"] / total * 100) if total > 0 else 0.0
    
    # Calculate profit factor
    winning_pnl = sum(
        (t["price"] - _active_positions.get(t["symbol"], {}).get("avg_price", t["price"])) * t["quantity"]
        for t in _trade_history
        if t["side"] == "sell" and (t["price"] - _active_positions.get(t["symbol"], {}).get("avg_price", t["price"])) > 0
    )
    losing_pnl = abs(sum(
        (t["price"] - _active_positions.get(t["symbol"], {}).get("avg_price", t["price"])) * t["quantity"]
        for t in _trade_history
        if t["side"] == "sell" and (t["price"] - _active_positions.get(t["symbol"], {}).get("avg_price", t["price"])) < 0
    ))
    metrics["profit_factor"] = (winning_pnl / losing_pnl) if losing_pnl > 0 else float('inf')
    
    return metrics


def reset_daily_metrics() -> None:
    """Reset daily P&L (called at market open)."""
    global _performance_metrics
    _performance_metrics["daily_pnl"] = 0.0
    _performance_metrics["last_reset"] = datetime.now(UTC).isoformat()
    LOGGER.info("[TRADE-MONITOR] Daily metrics reset")


def register_websocket(websocket) -> None:
    """Register a WebSocket connection for live updates."""
    _websocket_subscribers.add(websocket)
    LOGGER.info(f"[TRADE-MONITOR] WebSocket registered ({len(_websocket_subscribers)} total)")


def unregister_websocket(websocket) -> None:
    """Unregister a WebSocket connection."""
    _websocket_subscribers.discard(websocket)
    LOGGER.info(f"[TRADE-MONITOR] WebSocket unregistered ({len(_websocket_subscribers)} remaining)")


async def _broadcast_trade_update(trade_data: dict) -> None:
    """Broadcast trade update to all WebSocket subscribers."""
    if not _websocket_subscribers:
        return
    
    message = json.dumps({
        "type": "trade_update",
        "data": trade_data,
        "metrics": get_performance_metrics(),
        "timestamp": datetime.now(UTC).isoformat()
    })
    
    dead_sockets = set()
    for ws in _websocket_subscribers:
        try:
            await ws.send_text(message)
        except Exception as e:
            LOGGER.warning(f"[TRADE-MONITOR] WebSocket send failed: {e}")
            dead_sockets.add(ws)
    
    # Remove dead connections
    for ws in dead_sockets:
        _websocket_subscribers.discard(ws)


async def broadcast_metrics_update() -> None:
    """Broadcast current metrics to all subscribers (called periodically)."""
    if not _websocket_subscribers:
        return
    
    message = json.dumps({
        "type": "metrics_update",
        "data": get_performance_metrics(),
        "positions": get_active_positions(),
        "timestamp": datetime.now(UTC).isoformat()
    })
    
    dead_sockets = set()
    for ws in _websocket_subscribers:
        try:
            await ws.send_text(message)
        except Exception as e:
            dead_sockets.add(ws)
    
    for ws in dead_sockets:
        _websocket_subscribers.discard(ws)


def get_dashboard_summary() -> dict[str, Any]:
    """Get comprehensive dashboard summary."""
    metrics = get_performance_metrics()
    positions = get_active_positions()
    recent_trades = get_trade_history(limit=10)
    
    return {
        "ok": True,
        "performance": metrics,
        "positions": positions,
        "recent_trades": recent_trades,
        "subscribers": len(_websocket_subscribers),
        "timestamp": datetime.now(UTC).isoformat()
    }
